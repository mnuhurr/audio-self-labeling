from pathlib import Path
from tqdm import tqdm

import argparse
import audiofile as af
import h5py
import librosa
import math
import numpy as np
import torch
import torchaudio

from common import read_yaml, init_log
from fsd_meta import read_tags


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    #parser.add_argument('--logdir', help='tensorboard logdir')
    #parser.add_argument('--ckpt', help='continue from checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('prepare-data', level=cfg.get('log_level', 'info'))

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    cache_dir.mkdir(exist_ok=True, parents=True)

    sample_rate = cfg.get('sample_rate', 32000)
    n_fft = cfg.get('n_fft', 2048)
    hop_length = cfg.get('hop_length', n_fft // 2)
    n_mels = cfg.get('n_mels', 64)
    f_min = cfg.get('f_min', 0)
    f_max = cfg.get('f_max', sample_rate / 2)
    min_length = cfg.get('min_length', 0)
    max_length = cfg.get('max_length')
    segment_length = cfg.get('segment_length')
    segment_hop = cfg.get('segment_hop')

    exts = ['.wav', '.mp3']

    min_segment_length = cfg.get('min_segment_length', segment_length // 2)

    logger.debug(f'{cache_dir=}')
    logger.info(f'{sample_rate=}, {n_fft=}, {hop_length=}, {n_mels=}')
    logger.info(f'{segment_length=}, {segment_hop=}')

    # for generating gt labels
    tags, labels = read_tags(cfg.get('fsd_meta_fn'))
    label_idx = {label: k for k, label in enumerate(labels)}

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max)

    data_fn = Path(cfg.get('mel_fn', 'mels.hdf5'))
    logger.info(f'serializing data to {data_fn}')
    data_fn.parent.mkdir(exist_ok=True, parents=True)

    audio_data_dirs = cfg.get('audio_data_dirs', [])
    filenames = []
    for data_dir in audio_data_dirs:
        for ext in exts:
            filenames.extend(sorted(Path(data_dir).glob(f'*{ext}')))

    n_files = len(filenames)
    logger.info(f'found {n_files} files')

    max_sz = 10_000_000

    gt_rows = []

    with h5py.File(data_fn, 'w') as h5_file:
        dataset = h5_file.create_dataset(
            'mels',
            (max_sz, n_mels, segment_length),
            maxshape=(max_sz, n_mels, segment_length),
            chunks=(1, n_mels, segment_length))

        fname = h5_file.create_dataset(
            'fname',
            (max_sz,),
            maxshape=(10_000_000,),
            chunks=(1,),
            dtype=h5py.string_dtype(encoding='utf-8'))

        idx = 0

        for fn in tqdm(filenames):
            try:
                y, _ = librosa.load(fn, sr=sample_rate, mono=True)
            except OSError as e:
                print(f'error loading {fn} (duration {af.duration(fn):.1f} s): {e}')
                continue

            if len(y) / sample_rate < min_length:
                continue

            elif max_length is not None and len(y) / sample_rate > max_length:
                y = y[:int(max_length * sample_rate)]

            y = torch.as_tensor(y, dtype=torch.float32)
            mels = melspec(y)
            n_segments = math.ceil((mels.size(1) - segment_length) / segment_hop) + 1

            for k in range(n_segments):
                n = k * segment_hop
                seg = mels[:, n:n + segment_length].clone()
                if seg.size(-1) < min_segment_length:
                    continue

                if seg.size(-1) < segment_length:
                    missing = segment_length - seg.size(-1)
                    seg = torch.nn.functional.pad(seg, (0, missing))

                y_true = np.zeros(len(labels))
                for mid in tags[fn.stem]:
                    j = label_idx[mid]
                    y_true[j] = 1
                gt_rows.append(y_true)

                dataset[idx] = seg.numpy()
                fname[idx] = fn.stem
                idx += 1

        # done. resize back:
        dataset.resize((idx, n_mels, segment_length))
        fname.resize((idx,))

    logger.info(f'wrote {idx} mels')

    gt = np.stack(gt_rows)
    np.save(cache_dir / 'gt.npy', gt)


if __name__ == '__main__':
    main()
