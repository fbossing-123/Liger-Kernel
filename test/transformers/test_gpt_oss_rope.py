import pytest
import torch

from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gpt_oss
from liger_kernel.transformers.rope import liger_gpt_oss_rotary_pos_emb
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize("full_width", [False, True])
@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
@pytest.mark.parametrize("broadcast_batch", [False, True])
def test_gpt_oss_rope_forward_backward(full_width, unsqueeze_dim, broadcast_batch):
    q = torch.randn(2, 7, 4, 16, device=device)
    k = torch.randn(2, 7, 2, 16, device=device)
    if unsqueeze_dim == 1:
        q, k = q.transpose(1, 2), k.transpose(1, 2)
    q, k = q.requires_grad_(), k.requires_grad_()
    q_ref, k_ref = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    angles = torch.randn(1 if broadcast_batch else 2, 7, 8, device=device)
    cos, sin = 1.1 * angles.cos(), 1.1 * angles.sin()

    def reference(x):
        first, second = x.chunk(2, dim=-1)
        c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        return torch.cat((first * c - second * s, second * c + first * s), dim=-1)

    expected = reference(q_ref), reference(k_ref)
    if full_width:
        cos, sin = torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
    actual = liger_gpt_oss_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-5)
    dq, dk = torch.randn_like(q), torch.randn_like(k)
    expected_grads = torch.autograd.grad(expected, (q_ref, k_ref), (dq.clone(), dk.clone()))
    actual_grads = torch.autograd.grad(actual, (q, k), (dq.clone(), dk.clone()))
    for result, target in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-5)


def test_gpt_oss_rope_patch_matches_transformers(monkeypatch):
    modeling = pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
    config = modeling.GptOssConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    q, k = torch.randn(2, 4, 7, 16, device=device), torch.randn(2, 2, 7, 16, device=device)
    rotary = modeling.GptOssRotaryEmbedding(config).to(device)
    cos, sin = rotary(k, torch.arange(7, device=device).unsqueeze(0))
    expected = modeling.apply_rotary_pos_emb(q, k, cos, sin)
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", modeling.apply_rotary_pos_emb)
    apply_liger_kernel_to_gpt_oss(rms_norm=False, fused_linear_cross_entropy=False)
    assert modeling.apply_rotary_pos_emb is liger_gpt_oss_rotary_pos_emb
    actual = modeling.apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-5)
