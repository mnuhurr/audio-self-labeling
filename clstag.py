from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
import librosa
import audiofile as af

from typing import Dict, List, Optional, Tuple


def load_audio(filename: Path, sample_rate: int) -> torch.Tensor:
    y, _ = librosa.load(filename, sr=sample_rate, mono=True)

    return torch.as_tensor(y)


def cluster(model,
            filenames: List[Path],
            n_classes: List[int],
            melspec: torchaudio.transforms.MelSpectrogram,
            sample_rate: int,
            data_norm_params: dict[str, float] | None = None,
            min_length: float = 0.0,
            return_distributions: bool = False,
            device: Optional[torch.device] = None) -> Tuple[Dict[str, List[int]], Dict[str, torch.Tensor]]:
    device = device if device is not None else torch.device('cpu')

    max_cpu_length = 60

    classes = {}
    embeddings = {}

    data_mean = data_norm_params.get('mean', 0.0) if data_norm_params is not None else 0.0
    data_std = data_norm_params.get('std', 1.0) if data_norm_params is not None else 1.0

    for fn in tqdm(filenames):
        """
        try:
            y, sr = torchaudio.load(fn)
        except RuntimeError:
            continue

        if y.size(-1) / sr < min_length:
            continue

        y = y.mean(dim=0)

        if sr != sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sample_rate)

        """
        try:
            y = load_audio(fn, sample_rate=sample_rate)
        except OSError:
            print(f'error loading {fn} (duration {af.duration(fn):.2f} s)')
            continue

        dur = y.size(-1) / sample_rate

        if dur < min_length:
            continue

        mels = melspec(y)
        x = torch.log(mels + torch.finfo(mels.dtype).eps)
        if x.size(1) < 10:
            continue

        x = (x - data_mean) / data_std

        if dur < max_cpu_length:
            x = x.unsqueeze(0).to(device)
            y_pred, y_emb = model.clusters_and_clip_embedding(x)
        else:
            model = model.to('cpu')
            y_pred, y_emb = model.clusters_and_clip_embedding(x.unsqueeze(0))
            model = model.to(device)

        if return_distributions:
            classes[fn.stem] = [yp[0].cpu() for yp in y_pred]
        else:
            c = [torch.argmax(yp[0].cpu(), dim=-1).item() for yp in y_pred]
            classes[fn.stem] = c
        embeddings[fn.stem] = y_emb[0].cpu()

    return classes, embeddings
