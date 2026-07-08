import pytest
import torch

from eap.utils import tokenize_plus
from model_audit_helpers import (
    load_bridge_model,
    require_case_access,
    require_enabled,
    resolve_device,
    run_smoke_attribute_and_baseline,
    validate_prepared_model,
)
from model_audit_matrix import SMOKE_CASES, select_cases


pytestmark = [
    pytest.mark.integration,
    pytest.mark.model_audit_smoke,
]


REQUIRED_LAYER0_HOOKS = [
    "hook_embed",
    "blocks.0.hook_q_input",
    "blocks.0.hook_k_input",
    "blocks.0.hook_v_input",
    "blocks.0.attn.hook_result",
    "blocks.0.hook_mlp_in",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
]


def _capture_required_hook_shapes(model, tokens, attention_mask):
    shapes = {}

    def make_hook(name):
        def hook_fn(activations, hook):
            shapes[name] = tuple(activations.shape)

        return hook_fn

    hooks = [(name, make_hook(name)) for name in REQUIRED_LAYER0_HOOKS]
    with model.hooks(fwd_hooks=hooks):
        model(tokens, attention_mask=attention_mask)
    return shapes


def _capture_required_hook_grads(model, tokens, attention_mask, input_lengths):
    grads = {}

    def make_hook(name):
        def hook_fn(grad, hook):
            grads[name] = grad.detach().cpu()

        return hook_fn

    hooks = [(name, make_hook(name)) for name in REQUIRED_LAYER0_HOOKS]
    model.zero_grad()
    with model.hooks(bwd_hooks=hooks):
        logits = model(tokens, attention_mask=attention_mask)
        batch = torch.arange(logits.size(0), device=logits.device)
        final_pos = input_lengths.to(logits.device) - 1
        logits[batch, final_pos, 0].sum().backward()
    return grads


@pytest.mark.parametrize("case", select_cases(SMOKE_CASES), ids=lambda case: case.case_id)
def test_bridge_model_audit_smoke(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_SMOKE",
        "Set EAP_RUN_MODEL_AUDIT_SMOKE=1 to run model-audit smoke tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cpu")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)
    graph = run_smoke_attribute_and_baseline(model)

    clean = ["The cat sat on the mat"]
    tokens, attention_mask, input_lengths, _ = tokenize_plus(model, clean)
    model_device = next(model.parameters()).device
    tokens = tokens.to(model_device)
    attention_mask = attention_mask.to(model_device)

    shapes = _capture_required_hook_shapes(model, tokens, attention_mask)
    assert shapes.keys() == set(REQUIRED_LAYER0_HOOKS)
    assert all(all(dim > 0 for dim in shape) for shape in shapes.values())

    grads = _capture_required_hook_grads(model, tokens, attention_mask, input_lengths)
    assert grads.keys() == set(REQUIRED_LAYER0_HOOKS)
    assert all(torch.isfinite(value).all() for value in grads.values())

    assert graph.scores.shape == (graph.n_forward, graph.n_backward)
    assert graph.scores.abs().sum().item() > 0
