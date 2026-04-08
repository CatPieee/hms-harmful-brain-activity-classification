import torch
import torch.nn.functional as F


def kl_divergence_from_logits(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return F.kl_div(logp, target_prob, reduction="batchmean")


def accuracy_from_logits(logits: torch.Tensor, target_prob: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    true = torch.argmax(target_prob, dim=1)
    return float((pred == true).float().mean().item())
