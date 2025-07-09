from pathlib import Path

import argparse
import torch
import torchaudio

from clstag import cluster
from common import read_yaml, init_log
#from models.convnextish import ConvNeXtish, ConvNeXtConfig
from models.resnet import MultiheadResnet18
from models.vgglike import VGGLike


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    parser.add_argument('-d', '--distributions', action='store_true', default=False, help='store complete class distributions instead of class labels')
    parser.add_argument('-s', '--split', default='eval', help='dataset split')
    parser.add_argument('--ckpt', default=None, help='model state dict file')

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('cluster-tags', level=cfg.get('log_level', 'info'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    fsd50k_dir = Path(cfg.get('fsd50k_dir'))

    audio_dir = fsd50k_dir / f'FSD50K.{args.split}_audio'
    filenames = sorted(audio_dir.glob('*.wav'))

    # for output filenames
    prefix = f'fsd_{args.split}'

    sample_rate = cfg.get('sample_rate', 32000)
    n_fft = cfg.get('n_fft', 2048)
    hop_length = cfg.get('hop_length', n_fft // 2)
    n_mels = cfg.get('n_mels', 64)
    f_min = cfg.get('f_min', 0)
    f_max = cfg.get('f_max', sample_rate / 2)
    min_length = cfg.get('min_length', 0)
    logger.info(f'{min_length=}')

    #n_classes = cfg.get('n_classes', 1000)
    #n_channels = cfg.get('n_channels', 64)
    #depths = cfg.get('depths', [3, 3, 9, 3])
    #stem_factor = cfg.get('stem_factor', 4)
    #dropout = cfg.get('dropout', 0.2)
    #drop_path = cfg.get('drop_path', 0.0)
    n_heads = cfg.get('n_heads')
    n_classes_per_head = cfg.get('n_classes_per_head')
    n_classes = [n_classes_per_head] * n_heads

    model_path = args.ckpt if args.ckpt is not None else cfg.get('eval_model')

    data_norm_params = cfg.get('data_norm_params')

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max)

    """
    model_cfg = ConvNeXtConfig(
        n_channels=n_channels,
        n_classes=n_classes,
        depths=depths,
        stem_factor=stem_factor,
        dropout=dropout,
        drop_path=drop_path)

    model = ConvNeXtish(model_cfg).to(device)
    """
    model = MultiheadResnet18(n_classes=n_classes)
    #model = VGGLike(n_classes=n_classes)

    logger.info(f'loading model weights from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    classes, embeddings = cluster(
        model=model,
        filenames=filenames,
        n_classes=n_classes,
        melspec=melspec,
        sample_rate=sample_rate,
        return_distributions=args.distributions,
        data_norm_params=data_norm_params,
        min_length=min_length,
        device=device)

    torch.save(classes, f'data/{prefix}_classes.pt')
    torch.save(embeddings, f'data/{prefix}_embeddings.pt')


if __name__ == '__main__':
    main()
