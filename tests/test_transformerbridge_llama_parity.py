import copy
import os

import pytest
import torch

from eap.attribute import attribute
from eap.graph import Graph
from eap.model_adapter import prepare_model_for_eap
from eap.utils import make_hooks_and_matrices, tokenize_plus
from conftest import hf_or_skip, tl_parity_device
from parity_assertions import assert_strict_close


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("EAP_RUN_TL_PARITY") != "1",
        reason="Set EAP_RUN_TL_PARITY=1 to run real TransformerLens parity tests.",
    ),
]

LLAMA_ROPE_BASE = 500000.0
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B"
LLAMA_HOOK_VALUE_ATOL = 2e-6
LLAMA_CAPTURE_ATOL = 5e-6
LLAMA_HOOK_GRAD_ATOL = 2e-6
LLAMA_SCORE_ATOL = 2e-6


def _llama_snapshot() -> str:
    return LLAMA_MODEL_NAME


def _metric(logits, clean_logits, input_lengths, label):
    batch = torch.arange(logits.size(0), device=logits.device)
    final_pos = input_lengths.to(logits.device) - 1
    return logits[batch, final_pos, 0].sum()


def _tiny_dataloader():
    return [(["The cat sat on the mat"], ["The dog sat on the mat"], torch.tensor([0]))]


def _tiny_batch():
    clean, corrupted, label = _tiny_dataloader()[0]
    return clean, corrupted, label


def _required_hook_names(n_layers: int) -> list[str]:
    names = ["hook_embed"]
    for layer in range(n_layers):
        names.extend(
            [
                f"blocks.{layer}.hook_q_input",
                f"blocks.{layer}.hook_k_input",
                f"blocks.{layer}.hook_v_input",
                f"blocks.{layer}.attn.hook_result",
                f"blocks.{layer}.hook_mlp_in",
                f"blocks.{layer}.hook_mlp_out",
                f"blocks.{layer}.hook_resid_post",
            ]
        )
    return names


def _make_tokenizer(snapshot: str):
    transformers = pytest.importorskip("transformers")
    tokenizer = hf_or_skip(
        f"{snapshot} tokenizer",
        transformers.AutoTokenizer.from_pretrained,
        snapshot,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.padding_side = "right"
    return tokenizer


def _make_hf_model(tokenizer):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        rope_theta=LLAMA_ROPE_BASE,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
    )
    model = transformers.LlamaForCausalLM(config)
    model.eval()
    return model


def _make_base_state(tokenizer):
    torch.manual_seed(0)
    model = _make_hf_model(tokenizer)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _load_hf_model_from_state(tokenizer, state_dict):
    model = _make_hf_model(tokenizer)
    model.load_state_dict(state_dict)
    model.to(tl_parity_device())
    model.eval()
    return model


def _load_hooked_transformer(tokenizer, state_dict):
    transformer_lens = pytest.importorskip("transformer_lens")
    try:
        from transformer_lens.config import HookedTransformerConfig
        from transformer_lens.pretrained.weight_conversions.llama import convert_llama_weights
    except ImportError:
        pytest.skip("Local TransformerLens install does not expose Llama conversion helpers.")

    hf_model = _load_hf_model_from_state(tokenizer, state_dict)
    cfg_dict = {
        "d_model": hf_model.config.hidden_size,
        "d_head": hf_model.config.hidden_size // hf_model.config.num_attention_heads,
        "n_heads": hf_model.config.num_attention_heads,
        "d_mlp": hf_model.config.intermediate_size,
        "n_layers": hf_model.config.num_hidden_layers,
        "n_ctx": hf_model.config.max_position_embeddings,
        "eps": hf_model.config.rms_norm_eps,
        "d_vocab": hf_model.config.vocab_size,
        "act_fn": hf_model.config.hidden_act,
        "n_key_value_heads": None,
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_adjacent_pairs": False,
        "rotary_dim": hf_model.config.hidden_size // hf_model.config.num_attention_heads,
        "rotary_base": LLAMA_ROPE_BASE,
        "final_rms": True,
        "gated_mlp": True,
        "dtype": torch.float32,
        "device": tl_parity_device(),
        "n_devices": 1,
        "model_name": "tiny-llama-local",
        "original_architecture": "LlamaForCausalLM",
        "tokenizer_name": LLAMA_MODEL_NAME,
        "default_prepend_bos": True,
        "init_weights": False,
    }
    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    state_dict = convert_llama_weights(hf_model, cfg)
    model = transformer_lens.HookedTransformer(
        cfg,
        tokenizer=copy.deepcopy(tokenizer),
        move_to_device=False,
    )
    model.load_and_process_state_dict(state_dict)
    model.to(tl_parity_device())
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def _load_transformer_bridge(tokenizer, state_dict):
    pytest.importorskip("transformer_lens")
    try:
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError:
        pytest.skip("TransformerBridge is unavailable in this TransformerLens install.")

    hf_model = _load_hf_model_from_state(tokenizer, state_dict)
    bridge = hf_or_skip(
        f"{LLAMA_MODEL_NAME} TransformerBridge",
        TransformerBridge.boot_transformers,
        LLAMA_MODEL_NAME,
        hf_model=hf_model,
        tokenizer=copy.deepcopy(tokenizer),
        device=tl_parity_device(),
    )
    bridge = prepare_model_for_eap(bridge)
    bridge.eval()
    return bridge


def _capture_clean_prompt_backward_grads(model, inputs):
    tokenization = _capture_tokenization(model, inputs)
    tokens = tokenization["tokens"].to(next(model.parameters()).device)
    attention_mask = tokenization["attention_mask"].to(next(model.parameters()).device)
    input_lengths = tokenization["input_lengths"].to(next(model.parameters()).device)
    grads = {}

    def make_hook(name):
        def hook_fn(grad, hook):
            grads[name] = grad.detach().cpu()

        return hook_fn

    hooks = [(name, make_hook(name)) for name in _required_hook_names(model.cfg.n_layers)]
    model.zero_grad()
    with model.hooks(bwd_hooks=hooks):
        logits = model(tokens, attention_mask=attention_mask)
        batch = torch.arange(logits.size(0), device=logits.device)
        final_pos = input_lengths - 1
        logits[batch, final_pos, 0].sum().backward()
    return tokenization, grads


def _capture_hook_values(model, tokens, attention_mask=None):
    values = {}
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    def make_hook(name):
        def hook_fn(activations, hook):
            values[name] = activations.detach().cpu()

        return hook_fn

    hooks = [(name, make_hook(name)) for name in _required_hook_names(model.cfg.n_layers)]
    with model.hooks(fwd_hooks=hooks):
        model(tokens, attention_mask=attention_mask)
    return values


def _capture_tokenization(model, inputs):
    tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, inputs)
    return {
        "tokens": tokens.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
        "input_lengths": input_lengths.detach().cpu(),
        "n_pos": n_pos,
    }


def _capture_eap_ig_capture_stage(model, clean, corrupted):
    graph = Graph.from_model(model)
    clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
    corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)
    assert n_pos == n_pos_corrupted

    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=next(model.parameters()).device,
        dtype=model.cfg.dtype,
    )
    (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
        model,
        graph,
        batch_size=len(clean),
        n_pos=n_pos,
        scores=scores,
    )
    input_index = graph.forward_index(graph.nodes["input"])

    with torch.inference_mode():
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)
        activation_difference_corrupted = activation_difference.clone()
        input_activations_corrupted = activation_difference[:, :, input_index].clone()

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        activation_difference_final = activation_difference.clone()
        input_activations_clean = (
            input_activations_corrupted - activation_difference[:, :, input_index]
        ).clone()

    return {
        "clean_tokens": clean_tokens.detach().cpu(),
        "corrupted_tokens": corrupted_tokens.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
        "input_lengths": input_lengths.detach().cpu(),
        "n_pos": n_pos,
        "activation_difference_corrupted": activation_difference_corrupted.detach().cpu(),
        "activation_difference_final": activation_difference_final.detach().cpu(),
        "input_activations_corrupted": input_activations_corrupted.detach().cpu(),
        "input_activations_clean": input_activations_clean.detach().cpu(),
        "clean_logits": clean_logits.detach().cpu(),
    }


def _gradient_difference_summary(reference_grads, candidate_grads, threshold=1e-3):
    summary = []
    for name in sorted(reference_grads):
        diff = (candidate_grads[name] - reference_grads[name]).abs()
        summary.append(
            {
                "name": name,
                "max_abs_diff": diff.max().item(),
                "mean_abs_diff": diff.mean().item(),
                "count_gt_threshold": int((diff > threshold).sum().item()),
            }
        )
    summary.sort(key=lambda item: (-item["max_abs_diff"], -item["mean_abs_diff"], item["name"]))
    return summary


def _capture_non_permanent_hooks(model, *, dir="both"):
    return {
        name: len(handles)
        for name, handles in model.list_hooks(
            name_filter=_required_hook_names(model.cfg.n_layers),
            dir=dir,
            including_permanent=False,
        ).items()
    }


@pytest.fixture(scope="module")
def tiny_llama_artifacts():
    snapshot = _llama_snapshot()
    tokenizer = _make_tokenizer(snapshot)
    base_state = _make_base_state(tokenizer)
    return snapshot, tokenizer, base_state


def test_tiny_llama_tokenization_matches_between_hooked_transformer_and_bridge(
    tiny_llama_artifacts,
):
    _, tokenizer, base_state = tiny_llama_artifacts
    hooked = _load_hooked_transformer(tokenizer, base_state)
    bridge = _load_transformer_bridge(tokenizer, base_state)
    clean, _, _ = _tiny_batch()

    hooked_tokenization = _capture_tokenization(hooked, clean)
    bridge_tokenization = _capture_tokenization(bridge, clean)

    assert hooked_tokenization["n_pos"] == bridge_tokenization["n_pos"]
    torch.testing.assert_close(bridge_tokenization["tokens"], hooked_tokenization["tokens"])
    torch.testing.assert_close(
        bridge_tokenization["attention_mask"],
        hooked_tokenization["attention_mask"],
    )
    torch.testing.assert_close(
        bridge_tokenization["input_lengths"],
        hooked_tokenization["input_lengths"],
    )


def test_tiny_llama_hook_values_match_between_hooked_transformer_and_bridge(
    tiny_llama_artifacts,
):
    _, tokenizer, base_state = tiny_llama_artifacts
    hooked = _load_hooked_transformer(tokenizer, base_state)
    bridge = _load_transformer_bridge(tokenizer, base_state)
    clean, _, _ = _tiny_batch()

    tokenization = _capture_tokenization(hooked, clean)
    tokens = tokenization["tokens"]
    attention_mask = tokenization["attention_mask"]

    hooked_values = _capture_hook_values(hooked, tokens, attention_mask=attention_mask)
    bridge_values = _capture_hook_values(bridge, tokens, attention_mask=attention_mask)

    assert hooked_values.keys() == bridge_values.keys()
    for name in _required_hook_names(hooked.cfg.n_layers):
        assert_strict_close(
            bridge_values[name],
            hooked_values[name],
            atol=LLAMA_HOOK_VALUE_ATOL,
            name=f"Tiny-Llama hook value {name}",
        )


def test_tiny_llama_capture_stage_matches_between_hooked_transformer_and_bridge(
    tiny_llama_artifacts,
):
    _, tokenizer, base_state = tiny_llama_artifacts
    hooked = _load_hooked_transformer(tokenizer, base_state)
    bridge = _load_transformer_bridge(tokenizer, base_state)
    clean, corrupted, _ = _tiny_batch()

    hooked_state = _capture_eap_ig_capture_stage(hooked, clean, corrupted)
    bridge_state = _capture_eap_ig_capture_stage(bridge, clean, corrupted)

    assert hooked_state["n_pos"] == bridge_state["n_pos"]
    for key in ("clean_tokens", "corrupted_tokens", "attention_mask", "input_lengths"):
        torch.testing.assert_close(bridge_state[key], hooked_state[key])

    for key in (
        "activation_difference_corrupted",
        "activation_difference_final",
        "input_activations_corrupted",
        "input_activations_clean",
    ):
        assert_strict_close(
            bridge_state[key],
            hooked_state[key],
            atol=LLAMA_CAPTURE_ATOL,
            name=f"Tiny-Llama capture state {key}",
        )

    assert_strict_close(
        bridge_state["clean_logits"],
        hooked_state["clean_logits"],
        atol=LLAMA_CAPTURE_ATOL,
        name="Tiny-Llama capture clean logits",
    )


def test_tiny_llama_plain_backward_hook_gradients_match_between_hooked_transformer_and_bridge(
    tiny_llama_artifacts,
):
    _, tokenizer, base_state = tiny_llama_artifacts
    hooked = _load_hooked_transformer(tokenizer, base_state)
    bridge = _load_transformer_bridge(tokenizer, base_state)
    clean, _, _ = _tiny_batch()

    hooked_tokenization, hooked_grads = _capture_clean_prompt_backward_grads(hooked, clean)
    bridge_tokenization, bridge_grads = _capture_clean_prompt_backward_grads(bridge, clean)

    torch.testing.assert_close(bridge_tokenization["tokens"], hooked_tokenization["tokens"])
    torch.testing.assert_close(
        bridge_tokenization["attention_mask"],
        hooked_tokenization["attention_mask"],
    )
    torch.testing.assert_close(
        bridge_tokenization["input_lengths"],
        hooked_tokenization["input_lengths"],
    )
    assert hooked_grads.keys() == bridge_grads.keys()

    for name in _required_hook_names(hooked.cfg.n_layers):
        assert_strict_close(
            bridge_grads[name],
            hooked_grads[name],
            atol=LLAMA_HOOK_GRAD_ATOL,
            name=f"Tiny-Llama hook gradient {name}",
        )


def test_tiny_llama_bridge_does_not_leak_backward_hooks_after_plain_backward(
    tiny_llama_artifacts,
):
    _, tokenizer, base_state = tiny_llama_artifacts
    hooked = _load_hooked_transformer(tokenizer, base_state)
    bridge = _load_transformer_bridge(tokenizer, base_state)

    assert _capture_non_permanent_hooks(hooked, dir="bwd") == {}
    assert _capture_non_permanent_hooks(bridge, dir="bwd") == {}

    _ = _capture_clean_prompt_backward_grads(hooked, ["The cat sat on the mat"])
    _ = _capture_clean_prompt_backward_grads(bridge, ["The cat sat on the mat"])

    assert _capture_non_permanent_hooks(hooked, dir="bwd") == {}
    assert _capture_non_permanent_hooks(bridge, dir="bwd") == {}


def test_tiny_llama_eap_ig_scores_match_between_hooked_transformer_and_bridge(
    tiny_llama_artifacts,
):
    _, tokenizer, base_state = tiny_llama_artifacts
    hooked = _load_hooked_transformer(tokenizer, base_state)
    bridge = _load_transformer_bridge(tokenizer, base_state)

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
        atol=LLAMA_SCORE_ATOL,
        name="Tiny-Llama EAP-IG scores",
    )
