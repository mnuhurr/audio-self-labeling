
from pathlib import Path
import torch
import torch.nn.functional as F
import h5py


def collate_fn(batch: list[torch.Tensor]):
    batch_size = len(batch)
    n_mels = batch[0].size(0)
    maxlen = max(map(lambda x: x.size(-1), batch))

    x = torch.zeros(batch_size, n_mels, maxlen, dtype=batch[0].dtype)
    mask = torch.zeros(batch_size, maxlen)

    for k, item in enumerate(batch):
        item_len = item.size(-1)
        x[k, :, :item_len] = item
        mask[k, :item_len] = 1

    return x, mask


def collate(batch):
    mel = [item[0] for item in batch]
    c = [item[1] for item in batch]
    idx = [item[2] for item in batch]

    mel, _ = collate_fn(mel)
    return mel, torch.stack(c), torch.stack(idx)


class HDF5MelDataset(torch.utils.data.Dataset):
    def __init__(self, filename: Path,
                 output_length: int | None = None,
                 n_classes: list[int] = [10],
                 random_gain_low: float = 0.5, random_gain_high: float = 1.0,
                 random_stretch_range: float = 0.0,
                 noise_sigma: float = 0.0,
                 use_augmentation: float = True):

        self.data = h5py.File(filename, 'r')
        n_files = self.data['mels'].shape[0]

        self.classes = torch.zeros(n_files, len(n_classes), dtype=torch.int64)
        for k, nc in enumerate(n_classes):
            self.classes[:, k] = torch.arange(n_files) % nc

        # use data length as a backup: now all the outputs get fixed length
        self.output_length = output_length if output_length is not None else self.data['mels'].shape[-1]

        self.random_gain_low = random_gain_low
        self.random_gain_high = random_gain_high
        self.noise_sigma = noise_sigma
        self.stretch_range = random_stretch_range

        self.use_augmentation = use_augmentation

    def __len__(self) -> int:
        return self.data['mels'].shape[0]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(self.data['mels'][item])
        idx = torch.any(x != 0, dim=0)

        x[:, idx] = torch.log(x[:, idx] + torch.finfo(x.dtype).eps)

        if self.use_augmentation:
            x = self._random_stretch(x)
            x = self._random_gain(x)
            x = x + self.noise_sigma * torch.randn_like(x)

        x = self._fix_length(x)

        return x, self.classes[item], torch.tensor(item)

    def _random_gain(self, x: torch.Tensor) -> torch.Tensor:
        g = self.random_gain_low + (self.random_gain_high - self.random_gain_low) * torch.rand(size=())

        # assume x is log mels and converted using log_mels = log(mels + eps)
        return torch.log((torch.exp(x) - torch.finfo(x.dtype).eps) * g**2 + torch.finfo(x.dtype).eps)

    def _random_stretch(self, x: torch.Tensor) -> torch.Tensor:
        if self.stretch_range == 0:
            return x

        f = 2 * self.stretch_range * (torch.rand(size=()).item() - 0.5)
        x = F.interpolate(x.unsqueeze(0), scale_factor=(1 + f,))[0]
        return x

    def _fix_length(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < self.output_length:
            x = F.pad(x, (0, self.output_length - x.size(-1)))

        elif x.size(-1) > self.output_length:
            # limit the max offset by how much there is empty frames in the end of the data
            nonzeros = torch.any(x != 0.0, dim=0).int().sum().item()

            max_offset = min(nonzeros, x.size(-1) - self.output_length)
            offset = torch.randint(max_offset, size=())
            x = x[:, offset:offset + self.output_length]

        return x


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        return self.embeddings[item], item
