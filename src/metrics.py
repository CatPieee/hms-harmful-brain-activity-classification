import torch
import torch.nn.functional as F


def kl_divergence_from_logits(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    # Ensure targets are valid probabilities and avoid log(0)
    eps = 1e-6
    target_prob = target_prob.clamp(eps, 1.0)
    target_prob = target_prob / target_prob.sum(dim=1, keepdim=True)
    
    logp = F.log_softmax(logits, dim=1)
    # Using log_target=False: target * (log(target) - input)
    # We use a custom implementation to be safer with zeros if they still exist
    return F.kl_div(logp, target_prob.log(), reduction="batchmean", log_target=True)


def accuracy_from_logits(logits: torch.Tensor, target_prob: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    true = torch.argmax(target_prob, dim=1)
    return float((pred == true).float().mean().item())
