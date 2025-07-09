from datetime import date
from pathlib import Path

import argparse
import os
import torch

from common import read_yaml, init_log
from dataset import HDF5MelDataset, collate
from models.resnet import MultiheadResnet18
from models.utils import model_size
from trainer import Trainer, DDPTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='config file (default: settings.yaml)', default='settings.yaml')
    parser.add_argument('--checkpoint', help='checkpoint file to continue from', default=None)
    parser.add_argument('--logdir', help='log directory')
    parser.add_argument('--tensorboard', help='tensorboard output directory')

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_dir = args.logdir if args.logdir is not None else cfg.get('log_dir')
    log_fn = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        log_fn = log_dir / f'train-{str(date.today())}.log'
    logger = init_log('train', level=cfg.get('log_level', 'info'), filename=log_fn)

    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        local_rank = int(local_rank)

    if local_rank is not None:
        torch.distributed.init_process_group(backend='nccl')

    n_devices = torch.cuda.device_count()
    logger.info(f'found {n_devices} cuda devices')
    if local_rank is not None:
        torch.cuda.set_device(local_rank)

    mel_fn = Path(cfg.get('mel_fn', 'mels.hdf5'))

    if local_rank is None or local_rank == 0:
        logger.info(f'using serialized features from {mel_fn}')

    batch_size = cfg.get('batch_size', 16)
    num_workers = cfg.get('num_dataloader_workers', 0)
    n_epochs = cfg.get('n_epochs', 10)
    learning_rate = cfg.get('learning_rate', 0.05)
    weight_decay = cfg.get('weight_decay', 1e-5)
    clip_grad_norm = cfg.get('clip_grad_norm')
    mixup_alpha = cfg.get('mixup_alpha')
    log_interval = cfg.get('log_interval')
    n_splits = cfg.get('n_optimization_splits', 1)
    sk_lambda = cfg.get('sk_lambda', 25.0)
    save_every = cfg.get('save_every', 10)
    scheduler_step_size = cfg.get('scheduler_step_size', 5)
    scheduler_gamma = cfg.get('scheduler_gamma', 0.95)

    logger.info(f'{scheduler_step_size=} {scheduler_gamma=} {mixup_alpha=}')

    # model config specific thngs
    mcfg = cfg.get('model', {})

    dropout = mcfg.get('dropout', 0.1)
    n_heads = mcfg.get('n_heads', cfg.get('n_heads'))
    n_classes_per_head = mcfg.get('n_classes_per_head', cfg.get('n_classes_per_head'))
    n_classes = [n_classes_per_head] * n_heads

    n_time_mask = mcfg.get('n_time_mask', cfg.get('n_time_mask', 0))
    n_freq_mask = mcfg.get('n_freq_mask', cfg.get('n_freq_mask', 0))
    time_mask_param = mcfg.get('time_mask_param', cfg.get('time_mask_param', 50))
    freq_mask_param = mcfg.get('freq_mask_param', cfg.get('freq_mask_param', 16))

    train_segment_length = cfg.get('train_segment_length')
    noise_sigma = cfg.get('noise_sigma', 0.0)
    random_stretch_range = cfg.get('random_stretch_range', 0.0)
    if local_rank is None or local_rank == 0:
        logger.info(f'{train_segment_length=} {random_stretch_range=} {noise_sigma=}')

    models_dir = Path(cfg.get('models_dir', 'models'))
    models_dir.mkdir(exist_ok=True, parents=True)

    data_norm_params = cfg.get('data_norm_params')
    if data_norm_params is not None:
        if isinstance(data_norm_params, dict):
            data_norm_params = [
                data_norm_params['mean'],
                data_norm_params['std'],
            ]

        if local_rank is None or local_rank == 0:
            logger.info(f'{data_norm_params=}')

    train_ds = HDF5MelDataset(
        filename=mel_fn,
        n_classes=n_classes,
        output_length=train_segment_length,
        noise_sigma=noise_sigma,
        random_stretch_range=random_stretch_range)

    x, *_ = next(iter(train_ds))
    n_mels, n_mel_frames = x.shape

    if local_rank is None or local_rank == 0:
        logger.info(f'{len(train_ds)} files for training, {n_mels=}, {n_mel_frames=}')

    drop_last = mixup_alpha is not None

    if local_rank is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=torch.utils.data.DistributedSampler(train_ds),
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate)

    if local_rank is None or local_rank == 0:
        logger.info(f'batch_size={batch_size}')
        logger.info(f'using {n_splits} splits for prediction matrix in sinkhorn-knopp')

    optimize_on = [1, 2, 3, 4] + list(range(6, 61, 5)) + list(range(61, 121, 10)) + list(range(121, n_epochs, 20))
    optimize_on = [e for e in optimize_on if e < n_epochs]
    opt_epochs = map(str, sorted(optimize_on))

    if local_rank is None or local_rank == 0:
        logger.info('optimizing labels on epochs {}'.format(', '.join(opt_epochs)))

    model = MultiheadResnet18(
        n_classes=n_classes,
        dropout=dropout,
        data_norm_params=data_norm_params,
        n_time_mask=n_time_mask,
        n_freq_mask=n_freq_mask,
        time_mask_param=time_mask_param,
        freq_mask_param=freq_mask_param)

    if local_rank is None or local_rank == 0:
        logger.info(f'model size {model_size(model) / 1e6:.1f}M')

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    if local_rank is not None:
        trainer = DDPTrainer(
            gpu_id=local_rank,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            n_classes=n_classes,
            clip_grad_norm=clip_grad_norm,
            mixup_alpha=mixup_alpha,
            log_interval=log_interval,
            n_optimization_splits=n_splits,
            sinkhorn_knopp_lambda=sk_lambda,
            logger=logger if local_rank == 0 else None,
            save_every=save_every,
            checkpoint_dir=models_dir,
            optimize_on_epochs=optimize_on)
    else:
        trainer = Trainer(
            gpu_id=0,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            n_classes=n_classes,
            clip_grad_norm=clip_grad_norm,
            mixup_alpha=mixup_alpha,
            log_interval=log_interval,
            n_optimization_splits=n_splits,
            sinkhorn_knopp_lambda=sk_lambda,
            sinkhorn_knopp_tol=1e-4,
            use_stored_sk_vector=False,
            logger=logger,
            log_dir=args.tensorboard,
            save_every=save_every,
            checkpoint_dir=models_dir,
            optimize_on_epochs=optimize_on)

    if args.checkpoint is not None:
        logger.info(f'loading checkpoint from {args.checkpoint}')
        trainer.load_checkpoint(args.checkpoint)

    trainer.train(n_epochs)

    # save the final model
    torch.save(model.state_dict(), models_dir / f'model-{n_epochs}.pt')

    if local_rank is not None:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
