import torch
from typing import List


def get_preds_any(logits: torch.Tensor, lengths: torch.Tensor):
    # We should have a [batch, padded_len] tensor in logits, and a [batch] tensor in lengths
    items = []
    for idx in range(logits.shape[0]):
        idx_len = lengths[idx]
        items.append(torch.any(logits[idx][:idx_len] > 0).unsqueeze(0))
    return torch.cat(items)


def make_masked_shotmean_loss_fn(loss_fn):
    return lambda a, b, c: masked_shotmean_loss(loss_fn, a, b, c)


def masked_shotmean_loss(
    loss_fn, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor
):
    """Do a length-masked BCE loss, averaging within shots, then across the batch

    Args:
        loss_fn (_type_): a function (logits, labels) -> loss, like BCEWithLogitsLoss
        logits (torch.Tensor): [batch, padded_len] model logit output
        labels (torch.Tensor): [batch] labels of each shots
        lengths (torch.Tensor): [batch] lengths of each shot

    Returns:
        _type_: _description_
    """
    # Logits is [batch, padded_len], labels is [batch] lengths is [batch]
    losses = []
    for idx in range(logits.shape[0]):
        idx_len = lengths[idx]
        idx_loss = loss_fn(logits[idx][:idx_len], labels[idx].repeat(idx_len))
        losses.append(idx_loss.unsqueeze(0))

    return torch.mean(torch.cat(losses))


def test_get_preds_any():
    assert torch.all(
        get_preds_any(
            torch.tensor([[-5, -5, 1], [-2, 1, 1], [-5, -5, -5]]),
            torch.tensor([2, 2, 3]),
        )
        == torch.tensor([0, 1, 0])
    )


def test_masked_shotmean_loss():
    assert masked_shotmean_loss(
        torch.nn.BCEWithLogitsLoss(),
        torch.tensor([[-5.0, -5.0, 1.0], [-2.0, 1.0, 1.0], [-5.0, -5.0, -5.0]]),
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([2, 2, 3]),
    )
