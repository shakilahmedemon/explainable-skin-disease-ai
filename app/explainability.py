
import torch
import numpy as np

def compute_attention_rollout(model, input_tensor):
    """
    Compute ViT attention rollout.
    Returns 2D attention map (14x14).
    """
    model.eval()
    attentions = []

    def hook_fn(module, input, output):
        attentions.append(output.detach())

    hooks = []
    for block in model.model.blocks:
        hooks.append(block.attn.attn_drop.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    attn_mat = torch.stack(attentions).mean(dim=0)
    attn_mat = attn_mat[:, :, 1:, 1:]  # remove CLS token
    rollout = attn_mat.mean(dim=1)
    rollout = rollout.reshape(14, 14).cpu().numpy()
    rollout = rollout / rollout.max()

    return rollout
