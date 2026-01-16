import torch
import torch.nn.functional as F

def ensemble_predict(models, input_tensor):
    """
    Predict with deep ensemble and return mean probs, variance, entropy.
    """
    probs_list = []

    with torch.no_grad():
        for model in models:
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)

    probs_stack = torch.stack(probs_list)
    mean_probs = probs_stack.mean(dim=0)
    variance = probs_stack.var(dim=0).mean().item()
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1).item()

    return mean_probs.squeeze(), variance, entropy

