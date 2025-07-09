import time
import math
import torch
import torch.nn.functional as F

from typing import Optional, Tuple


#@torch.jit.script
@torch.inference_mode()
def optimize_single(pr: torch.Tensor,
                    lamb: float = 25.0,
                    tol: float = 1e-1,
                    target_class_probs: Optional[torch.Tensor] = None,
                    previous_r: Optional[torch.Tensor] = None,
                    previous_c: Optional[torch.Tensor] = None,
                    device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    n_files, n_classes = pr.shape

    r = torch.ones(n_classes, 1, dtype=torch.float64, device=device) / n_classes if previous_r is None else previous_r.to(device)
    c = torch.ones(n_files, 1, dtype=torch.float64, device=device) / n_files if previous_c is None else previous_c.to(device)
    pr = pr.T.pow_(lamb).to(device)

    if target_class_probs is None:
        # target_class_probs = torch.ones(n_classes, dtype=torch.float64, device=device) / n_classes
        target_class_probs = torch.tensor(1 / n_classes, dtype=torch.float64, device=device)
    elif target_class_probs.dim() == 1:
        target_class_probs = target_class_probs.unsqueeze(-1).to(torch.float64).to(device)

    #inv_k = 1 / n_classes
    inv_n = 1 / n_files

    #t0 = time.time()
    err = float('inf')
    count = 0
    while err > tol:
        r = target_class_probs / (pr @ c)
        c_new = inv_n / (r.T @ pr).T

        if count % 10 == 0:
            err = torch.nansum(torch.abs(c / c_new - 1))

        c = c_new
        count += 1

    #dt = time.time() - t0
    #if verbose:
    #    print(f'done in {count} rounds ({dt:.1f} s)')

    pr = pr.cpu()
    c = c.cpu()
    r = r.cpu()
    pr *= c.squeeze()
    pr = pr.T * r.squeeze()
    return pr, r.cpu(), c.cpu(), count


#@torch.jit.script
@torch.inference_mode()
def optimize_split(predictions: torch.Tensor, n_splits: int = 2, lamb: float = 25.0, tol: float = 1.0e-1,
                   previous_r: Optional[torch.Tensor] = None,
                   device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    sinkhorn knopp optimization but the matrix multiplication is divided into n_splits parts so that
    everything fits to a single gpu memory
    """

    n_files, n_classes = predictions.shape
    r = torch.ones(n_classes, 1, dtype=torch.float64, device=device) / n_classes if previous_r is None else previous_r.to(device)
    c = torch.ones(n_files, 1, dtype=torch.float64, device=device) / n_files if previous_c is None else previous_c.to(device)
    pr = predictions.T ** lamb

    split_size = math.ceil(n_files / n_splits)

    inv_k = torch.tensor(1 / n_classes, dtype=torch.float64, device=device)
    inv_n = torch.tensor(1 / n_files, dtype=torch.float64, device=device)

    #t0 = time.time()
    err = float('inf')
    count = 0
    while err > tol:
        c_new = torch.zeros_like(c)
        rt = torch.zeros_like(r)

        for s in range(n_splits):
            start = s * split_size
            end = (s + 1) * split_size
            pr_part = pr[:, start:end].to(device)
            rt += pr_part @ c[start:end]
        r = inv_k / rt
        # del rt

        for s in range(n_splits):
            start = s * split_size
            end = (s + 1) * split_size
            pr_part = pr[:, start:end].to(device)
            c_new[start:end, 0] = (inv_n / (r.T @ pr_part))[0]

        if count % 10 == 0:
            err = torch.nansum(torch.abs(c / c_new - 1))

        c = c_new
        count += 1

    #dt = time.time() - t0
    #if verbose:
    #    print(f'done in {count} rounds ({dt:.1f} s)')

    pr *= c.squeeze().cpu()
    pr = pr.T * r.squeeze().cpu()
    return pr, r.cpu(), c.cpu()


def main():
    import timeit
    n_classes = 2000
    n_files = 260000
    dev = torch.device('cpu')
    temp = 10

    def run_single():
        pr = torch.softmax(torch.randn(n_files, n_classes, dtype=torch.float64) / temp, dim=-1)
        optimize_single(pr, device=dev)

    def run_split(n_splits=10):
        pr = torch.softmax(torch.randn(n_files, n_classes, dtype=torch.float64) / temp, dim=-1)
        optimize_split(pr, n_splits=n_splits, device=dev)

    #pr[:, 1] = 1
    """
    pr_s, _ = optimize_single(pr.clone().to(dev))
    pr_h, _ = optimize_split(pr.clone(), n_splits=4, device=dev)

    print(f'difference for single: {F.mse_loss(pr_s, pr).item():.4f}')
    print(f'difference for splits: {F.mse_loss(pr_h, pr).item():.4f}')
    print(f'difference between opts: {F.mse_loss(pr_s, pr_h).item():.4f}')
    """
    n = 20
    t1 = timeit.timeit(run_single, number=n)
    t2 = timeit.timeit(run_split, number=n)

    print(f'single: {t1 / n:.1f} s')
    print(f'splits: {t2 / n:.1f} s')


if __name__ == '__main__':
    main()
