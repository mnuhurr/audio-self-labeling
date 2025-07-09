import torch
import torchvision

from .augmentations import RandomResizeCrop, RandomLinearFader
from .utils import SpecAugment



class MultiheadResnet18(torch.nn.Module):
    def __init__(self,
                 n_classes: list[int],
                 dropout: float = 0.0,
                 data_norm_params: tuple[float] | None = None,
                 n_time_mask: int = 0,
                 n_freq_mask: int = 0,
                 time_mask_param: int = 50,
                 freq_mask_param: int = 16):
        super().__init__()

        #self.scaling = MelScaling(n_mels)
        self.data_norm_params = data_norm_params

        # byol-a augmentations
        self.augmentations = torch.nn.ModuleList([
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
            RandomLinearFader(),
        ])

        self.specaugment = SpecAugment(
            n_time_mask=n_time_mask,
            n_freq_mask=n_freq_mask,
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param)


        resnet = torchvision.models.resnet18()

        # modify the first conv to take a single channel input instead of three color channels
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove final projection to classes
        resnet.fc = torch.nn.Identity()

        self.resnet = resnet
        self.head = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(512, c)
            ) for c in n_classes
        )

        self.apply(self._init_weight)

    def _init_weight(self, m: torch.nn.Module):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def clip_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_norm_params is not None:
            mean, std = self.data_norm_params
            x = (x - mean) / std
        # x = self.scaling(x)
        if self.training:
            for aug in self.augmentations:
                x = aug(x)

        x = self.specaugment(x)
        x = self.resnet(x.unsqueeze(1))
        return x

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.clip_embedding(x)
        y = [h(x) for h in self.head]
        return y

    def clusters_and_clip_embedding(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """return both, clip embeddings and the classifier outputs"""
        x_emb = self.clip_embedding(x)
        y = [h(x_emb) for h in self.head]
        return y, x_emb


def _test():
    from utils import model_size

    x = torch.randn(4, 64, 300)
    model = MultiheadResnet18([100, 100], n_mels=64)

    y = model(x)
    print([yy.shape for yy in y])
    print(model_size(model)/1e6)


if __name__ == '__main__':
    _test()
