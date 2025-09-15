import time
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from dataset import EmbeddingDataset
import sinkhorn_knopp as sk
from utils import cls_tag_mi

from typing import Any


@torch.no_grad()
def spectrogram_mixing(x: torch.Tensor, max_amount: float = 0.3) -> torch.Tensor:
    u = max_amount * torch.rand(x.size(0), 1, 1, device=x.device)
    # assume x is in log-space, and do the mixing in the exp-space
    xe = torch.exp(x)
    xm = (1 - u) * xe + u * torch.flip(xe, dims=(0,))
    return torch.log(xm)


class Trainer:
    def __init__(self,
                 device: torch.device,
                 model: torch.nn.Module,
                 loader: torch.utils.data.DataLoader,
                 n_classes: list[int],
                 optimizer: torch.optim.Optimizer,
                 optimize_on_epochs: list[int],
                 epoch: int | None = None,
                 scheduler: Any | None = None,
                 log_interval: int | None = None,
                 sinkhorn_knopp_lambda: float = 25.0,
                 sinkhorn_knopp_tol: float = 1e-1,
                 n_optimization_splits: int = 1,
                 clip_grad_norm: float | None = None,
                 gt: torch.Tensor | None = None,
                 mixup_alpha: float | None = None,
                 logger: logging.Logger | None = None,
                 checkpoint_dir: str | Path = 'checkpoints',
                 log_dir: str | Path | None = None,
                 use_stored_sk_vector: bool = False,
                 save_every: int = 10):
        self.device = device
        self.model = model.to(self.device)
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.log_interval = log_interval
        self.optimize_on = optimize_on_epochs
        self.kl_beta = 0.1

        self.n_classes = n_classes
        self.accuracy = [torchmetrics.Accuracy('multiclass', num_classes=nc).to(self.device) for nc in n_classes]

        self.scaler = torch.amp.GradScaler(device=self.device)

        self.epoch = epoch if epoch is not None else 0
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)

        # the input dimension of the last layer of the first head (which is a nn.Sequential)
        self.d_embedding = model.head[0][-1].weight.size(1)
        self.sk_lambda = sinkhorn_knopp_lambda
        self.sk_splits = n_optimization_splits
        self.sk_tol = sinkhorn_knopp_tol

        # sinkhorn-knopp r/c params from previous optimization
        self.sk_r = [None] * len(self.n_classes)
        self.sk_c = [None] * len(self.n_classes)
        self.store_sk = use_stored_sk_vector

        self.clip_grad_norm = clip_grad_norm
        self.mixup_alpha = mixup_alpha

        self.logger = logger
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
        }

        self.step = 0

        self.gt = gt

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)

    def _train_batch(self, x: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            y_pred = self.model(x)
            losses = torch.stack([self.criterion(yp, y_true[:, k]) for k, yp in enumerate(y_pred)])
            loss = losses.mean()

        self.scaler.scale(loss).backward()

        if self.clip_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()

        accs = []
        with torch.no_grad():
            #if self.mixup_alpha is not None:
            #    y_true = torch.argmax(y_true, dim=-1)

            for k, yp in enumerate(y_pred):
                self.accuracy[k].update(yp, y_true[:, k])
                accs.append((torch.argmax(yp, dim=-1) == y_true[:, k]).float().mean())
            batch_acc = torch.stack(accs).mean().item()

        return losses.detach().cpu(), batch_acc

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        loss = torch.zeros(len(self.n_classes))
        batch_t0 = time.time()

        for batch, (x, y_true, _) in enumerate(self.loader):
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            if self.mixup_alpha is not None:
                #x, y_true = batch_mixup(x, y_true, n_classes=self.n_classes[0], mixup_alpha=self.mixup_alpha)
                x = spectrogram_mixing(x, max_amount=self.mixup_alpha)

            batch_loss, batch_acc = self._train_batch(x, y_true)
            loss += batch_loss

            if self.log_interval is not None and batch % self.log_interval == 0:
                t_batch = (time.time() - batch_t0) * 1000 / self.log_interval

                avg_loss = batch_loss.mean().item()
                avg_acc = torch.stack([acc.compute() for acc in self.accuracy]).mean().item()

                current_lr = self.optimizer.param_groups[0]['lr']
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', avg_loss, self.step)
                    self.writer.add_scalar('train/accuracy', batch_acc, self.step)
                    self.writer.add_scalar('learning_rate', current_lr, self.step)

                print(f'batch {batch:5d}/{len(self.loader)} | {int(t_batch):4d} ms/batch | lr {current_lr:.3g} | training loss {avg_loss:.4f} | accuracy {batch_acc:.4f} (avg {avg_acc:.4f})')
                batch_t0 = time.time()

            self.step += 1

        loss = loss / len(self.loader)

        # collect data & clean up
        accuracy = torch.stack([acc.compute() for acc in self.accuracy])

        self.history['train_loss'].append(loss.detach().cpu())
        self.history['train_accuracy'].append(accuracy.detach().cpu())

        if self.writer is not None:
            losses = {}
            accuracies = {}
            for hn in range(len(self.n_classes)):
                losses[f'head_{hn}'] = loss[hn].item()
                accuracies[f'head_{hn}'] = accuracy[hn].item()

            self.writer.add_scalars('epoch/loss', losses, self.epoch + 1)
            self.writer.add_scalars('epoch/accuracy', accuracies, self.epoch + 1)
            self.writer.flush()

        epoch_loss = loss.mean().item()
        epoch_acc = accuracy.mean().item()

        for acc in self.accuracy:
            acc.reset()

        return epoch_loss, epoch_acc

    @torch.inference_mode()
    def _clip_embeddings(self) -> torch.Tensor:
        n = len(self.loader.dataset)
        embeddings = torch.empty(n, self.d_embedding, device=self.device)

        for x, _, idx in tqdm(self.loader):
            x = x.to(self.device)
            embeddings[idx] = self.model.clip_embedding(x)

        return embeddings

    @torch.inference_mode()
    def _optimize_labels(self, temperature: float = 1.0):
        self.model.eval()

        n_heads = len(self.n_classes)

        # 0. make sure there is no augmentions
        use_augmentation = self.loader.dataset.use_augmentation
        output_length = self.loader.dataset.output_length
        self.loader.dataset.use_augmentation = False
        self.loader.dataset.output_length = self.loader.dataset.data['mels'].shape[2]

        # 1. collect labels
        embeddings = self._clip_embeddings()
        n = len(self.loader.dataset)
        emb_ds = EmbeddingDataset(embeddings)
        emb_loader = torch.utils.data.DataLoader(emb_ds, batch_size=512)

        # 2. iteration
        times = torch.zeros(n_heads)
        scalars = {}
        counts = {}
        nmi = {}
        n_rnds = []
        for hn in tqdm(range(n_heads)):
            #print(f'head {hn + 1}: calculating projections... ', end='', flush=True)
            torch.cuda.empty_cache()
            predicted = torch.zeros(n, self.n_classes[hn], dtype=torch.float64, device=self.device)
            for x, idx in emb_loader:
                predicted[idx] = F.softmax(self.model.head[hn](x).to(torch.float64) / temperature, dim=-1)

            #print('optimizing... ', end='', flush=True)

            t0 = time.time()
            if self.sk_splits > 1:
                pr, r, c = sk.optimize_split(
                    predicted,
                    n_splits=self.sk_splits,
                    lamb=self.sk_lambda,
                    previous_r=self.sk_r[hn],
                    previous_c=self.sk_c[hn],
                    tol=self.sk_tol,
                    device=self.device)
            else:
                pr, r, c, cnt = sk.optimize_single(
                    predicted,
                    lamb=self.sk_lambda,
                    previous_r=self.sk_r[hn],
                    previous_c=self.sk_c[hn],
                    tol=self.sk_tol,
                    device=self.device)
                n_rnds.append(cnt)

            del predicted
            dt = time.time() - t0
            #print(f'done in {dt:.1f} s')
            times[hn] = dt
            scalars[f'head_{hn}'] = dt
            counts[f'head_{hn}'] = cnt

            if self.store_sk:
                self.sk_r[hn] = r.cpu()
                self.sk_c[hn] = c.cpu()

            torch.cuda.empty_cache()
            new_labels = torch.argmax(pr, dim=1).to(self.loader.dataset.classes.dtype)

            assert len(new_labels) == n
            self.loader.dataset.classes[:, hn] = new_labels

            # stats:
            if self.gt is not None:
                cls_mat = F.one_hot(new_labels, num_classes=pr.size(1))
                nmi[f'head_{hn}'] = cls_tag_mi(cls_mat, self.gt)

        if self.writer is not None:
            self.writer.add_scalars('optimize/times', scalars, self.epoch + 1)
            self.writer.add_scalars('optimize/iterations', counts, self.epoch + 1)
            if len(nmi) > 0:
                self.writer.add_scalars('optimize/nmi', nmi, self.epoch + 1)

            self.writer.flush()

        if len(nmi) > 0:
            avg_nmi = torch.mean(torch.as_tensor(list(nmi.values()))).item()
            nmi_str = f', avg nmi {avg_nmi:.4f}'
        else:
            nmi_str = ''

        if len(n_rnds) > 0:
            avg_rounds = torch.as_tensor(n_rnds, dtype=torch.float32).mean().item()
            cnt_str = f', avg rounds {avg_rounds:.1f}'
        else:
            cnt_str = ''

        print(f'average optimization time {times.mean().item():.1f} s{nmi_str}{cnt_str}')
        # restore
        self.loader.dataset.use_augmentation = use_augmentation
        self.loader.dataset.output_length = output_length

    def train(self, n_epochs: int):
        while self.epoch < n_epochs:
            if self.epoch + 1 in self.optimize_on:
                self._log(f'epoch {self.epoch + 1} - optimize labels')
                self._optimize_labels()

            self._log(f'epoch {self.epoch + 1} - train network')

            loss, acc = self._train_epoch()
            lr_str = ''

            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                current_lr = self.optimizer.param_groups[0]['lr']
                lr_str = f' learning rate {current_lr:.5g} -'
                self.scheduler.step()

            self._log(f'epoch {self.epoch + 1} -{lr_str} avg loss {loss:.4f} - avg accuracy {acc:.4f}')

            if (self.epoch + 1) % self.save_every == 0:
                self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
                self.save_checkpoint(self.checkpoint_dir / f'ckpt-{self.epoch + 1:03d}.pt')

            self.epoch += 1

    def save_checkpoint(self, filename: Path):
        ckpt_data = {
            'epoch': self.epoch,
            'optimize_on': self.optimize_on,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'classes': self.loader.dataset.classes,
            'history': self.history,
            'sk_r': self.sk_r,
            #'sk_c': self.sk_c,
        }

        torch.save(ckpt_data, filename)

    def load_checkpoint(self, filename):
        ckpt_data = torch.load(filename)

        self.epoch = ckpt_data['epoch'] + 1
        self.step = self.epoch * len(self.loader)
        self.optimize_on = ckpt_data['optimize_on']
        self.model.load_state_dict(ckpt_data['model'])
        self.optimizer.load_state_dict(ckpt_data['optimizer'])
        self.loader.dataset.classes = ckpt_data['classes']
        self.history = ckpt_data['history']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt_data['scheduler'])
        self.sk_r = ckpt_data['sk_r']
        #self.sk_c = ckpt_data['sk_c']
