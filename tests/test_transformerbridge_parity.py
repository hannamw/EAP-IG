import os

import pytest
import torch

from eap.attribute import attribute
from eap.graph import Graph
from eap.model_adapter import prepare_model_for_eap
from conftest import hf_or_skip, tl_parity_device
from parity_assertions import assert_strict_close


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("EAP_RUN_TL_PARITY") != "1",
        reason="Set EAP_RUN_TL_PARITY=1 to run real TransformerLens parity tests.",
    ),
]


REQUIRED_HOOK_NAMES = [
    "hook_embed",
    "blocks.0.hook_q_input",
    "blocks.0.hook_k_input",
    "blocks.0.hook_v_input",
    "blocks.0.attn.hook_result",
    "blocks.0.hook_mlp_in",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
]

GPT2_LOGITS_ATOL = 2e-5
GPT2_HOOK_VALUE_ATOL = 3e-5
GPT2_HOOK_GRAD_ATOL = 2e-6
GPT2_SCORE_ATOL = 5e-5


def _load_hooked_transformer():
    transformer_lens = pytest.importorskip("transformer_lens")
    hooked_cls = getattr(transformer_lens, "HookedTransformer", None)
    if hooked_cls is None:
        pytest.skip("HookedTransformer is unavailable in this TransformerLens install.")

    model = hf_or_skip(
        "GPT-2 HookedTransformer",
        hooked_cls.from_pretrained,
        "gpt2",
        device=tl_parity_device(),
    )
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    return model


def _load_transformer_bridge():
    pytest.importorskip("transformer_lens")
    try:
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError:
        pytest.skip("TransformerBridge is unavailable in this TransformerLens install.")

    bridge = hf_or_skip(
        "GPT-2 TransformerBridge",
        TransformerBridge.boot_transformers,
        "gpt2",
        device=tl_parity_device(),
    )
    return prepare_model_for_eap(bridge)


def _metric(logits, clean_logits, input_lengths, label):
    batch = torch.arange(logits.size(0), device=logits.device)
    final_pos = input_lengths.to(logits.device) - 1
    return logits[batch, final_pos, 0].sum()


def _tiny_dataloader():
    return [(["The cat sat on the mat"], ["The dog sat on the mat"], torch.tensor([0]))]


def _capture_hook_shapes(model, tokens):
    shapes = {}

    def make_hook(name):
        def hook_fn(activations, hook):
            shapes[name] = tuple(activations.shape)

        return hook_fn

    hooks = [(name, make_hook(name)) for name in REQUIRED_HOOK_NAMES]
    with model.hooks(fwd_hooks=hooks):
        model(tokens)
    return shapes


def _capture_hook_values(model, tokens):
    values = {}

    def make_hook(name):
        def hook_fn(activations, hook):
            values[name] = activations.detach().cpu()

        return hook_fn

    hooks = [(name, make_hook(name)) for name in REQUIRED_HOOK_NAMES]
    with model.hooks(fwd_hooks=hooks):
        model(tokens)
    return values


def _capture_hook_grads(model, tokens):
    grads = {}

    def make_hook(name):
        def hook_fn(grad, hook):
            grads[name] = grad.detach().cpu()

        return hook_fn

    hooks = [(name, make_hook(name)) for name in REQUIRED_HOOK_NAMES]
    model.zero_grad()
    with model.hooks(bwd_hooks=hooks):
        logits = model(tokens)
        logits[0, -1, 0].backward()
    return grads


def test_hooked_transformer_and_bridge_logits_match_in_compatibility_mode():
    hooked = _load_hooked_transformer()
    bridge = _load_transformer_bridge()
    tokens = hooked.to_tokens(["The cat sat on the mat"], prepend_bos=True)

    with torch.inference_mode():
        hooked_logits = hooked(tokens)
        bridge_logits = bridge(tokens)

    assert_strict_close(
        bridge_logits,
        hooked_logits,
        atol=GPT2_LOGITS_ATOL,
        name="GPT-2 logits",
    )


def test_required_eap_hook_shapes_match_between_hooked_transformer_and_bridge():
    hooked = _load_hooked_transformer()
    bridge = _load_transformer_bridge()
    tokens = hooked.to_tokens(["The cat sat on the mat"], prepend_bos=True)

    hooked_shapes = _capture_hook_shapes(hooked, tokens)
    bridge_shapes = _capture_hook_shapes(bridge, tokens)

    assert hooked_shapes.keys() == bridge_shapes.keys()
    assert bridge_shapes == hooked_shapes


def test_required_eap_hook_values_match_between_hooked_transformer_and_bridge():
    hooked = _load_hooked_transformer()
    bridge = _load_transformer_bridge()
    tokens = hooked.to_tokens(["The cat sat on the mat"], prepend_bos=True)

    hooked_values = _capture_hook_values(hooked, tokens)
    bridge_values = _capture_hook_values(bridge, tokens)

    assert hooked_values.keys() == bridge_values.keys()
    for name in REQUIRED_HOOK_NAMES:
        assert_strict_close(
            bridge_values[name],
            hooked_values[name],
            atol=GPT2_HOOK_VALUE_ATOL,
            name=f"GPT-2 hook value {name}",
        )


def test_required_eap_hook_gradients_match_between_hooked_transformer_and_bridge():
    hooked = _load_hooked_transformer()
    bridge = _load_transformer_bridge()
    tokens = hooked.to_tokens(["The cat sat on the mat"], prepend_bos=True)

    hooked_grads = _capture_hook_grads(hooked, tokens)
    bridge_grads = _capture_hook_grads(bridge, tokens)

    assert hooked_grads.keys() == bridge_grads.keys()
    for name in REQUIRED_HOOK_NAMES:
        assert_strict_close(
            bridge_grads[name],
            hooked_grads[name],
            atol=GPT2_HOOK_GRAD_ATOL,
            name=f"GPT-2 hook gradient {name}",
        )


def test_bridge_attribution_runs_with_legacy_compatible_hooks_by_default():
    bridge = _load_transformer_bridge()
    bridge_graph = Graph.from_model(bridge)

    attribute(
        bridge,
        bridge_graph,
        _tiny_dataloader(),
        _metric,
        method="EAP",
        quiet=True,
    )
    assert bridge_graph.scores.shape == (bridge_graph.n_forward, bridge_graph.n_backward)


def test_tiny_eap_scores_match_between_hooked_transformer_and_bridge():
    hooked = _load_hooked_transformer()
    bridge = _load_transformer_bridge()

    hooked_graph = Graph.from_model(hooked)
    bridge_graph = Graph.from_model(bridge)

    attribute(hooked, hooked_graph, _tiny_dataloader(), _metric, method="EAP", quiet=True)
    attribute(
        bridge,
        bridge_graph,
        _tiny_dataloader(),
        _metric,
        method="EAP",
        quiet=True,
    )

    assert_strict_close(
        bridge_graph.scores,
        hooked_graph.scores,
        atol=GPT2_SCORE_ATOL,
        name="GPT-2 EAP scores",
    )


def test_tiny_eap_ig_scores_match_between_hooked_transformer_and_bridge():
    hooked = _load_hooked_transformer()
    bridge = _load_transformer_bridge()

    hooked_graph = Graph.from_model(hooked)
    bridge_graph = Graph.from_model(bridge)

    attribute(
        hooked,
        hooked_graph,
        _tiny_dataloader(),
        _metric,
        method="EAP-IG-inputs",
        ig_steps=2,
        quiet=True,
    )
    attribute(
        bridge,
        bridge_graph,
        _tiny_dataloader(),
        _metric,
        method="EAP-IG-inputs",
        ig_steps=2,
        quiet=True,
    )

    assert_strict_close(
        bridge_graph.scores,
        hooked_graph.scores,
        atol=GPT2_SCORE_ATOL,
        name="GPT-2 EAP-IG scores",
    )
