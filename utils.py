
import torch



def combined_counts(pred_mat: torch.Tensor, gt_mat: torch.Tensor) -> torch.Tensor:
    # class/tag numbers
    n_pred = pred_mat.shape[1]
    n_true = gt_mat.shape[1]

    print(pred_mat.shape, gt_mat.shape)
    # they should have the same number of files/predicted items
    assert pred_mat.shape[0] == gt_mat.shape[0]
    n_files = pred_mat.shape[0]

    mat = torch.zeros(n_pred, n_true)

    for k in range(n_files):
        for i in torch.where(pred_mat[k, :] > 0):
            for j in torch.where(gt_mat[k, :] > 0):
                mat[i, j] += 1

    return mat


def cls_tag_mi(classes: torch.Tensor, tags: torch.Tensor, normalize: bool = True) -> float:
    """
     compare a class matrix against a tag matrix
    """
    cm = combined_counts(classes, tags)

    # for mi
    p_x = torch.sum(classes, axis=0) / torch.sum(classes)
    p_y = torch.sum(tags, axis=0) / torch.sum(tags)
    p_xy = cm / torch.sum(cm)

    ppxy = p_x[:, None] @ p_y[None, :]

    ind = (p_xy > 0) & (ppxy > 0)

    mi = torch.sum(p_xy[ind] * torch.log2(p_xy[ind] / ppxy[ind]))

    if normalize:
        mi = mi / torch.log2(torch.as_tensor(classes.shape[1]))

    return mi.item()


def _test():
    n_files = 100000
    n_cls = 1000
    n_tags = 200

    y_pred = torch.nn.functional.one_hot(
        torch.randint(n_cls, size=(n_files,)),
        num_classes=n_cls)

    y_tags = (torch.rand(n_files, n_tags) < 0.1).float()

    nmi = cls_tag_mi(y_pred, y_tags)
    print(nmi)


if __name__ == '__main__':
    _test()
            
