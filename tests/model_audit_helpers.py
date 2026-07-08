from __future__ import annotations

import os
import gc
from typing import Any

import pytest
import torch

from eap.attribute import attribute
from eap.evaluate import evaluate_baseline
from eap.graph import Graph
from eap.model_adapter import prepare_model_for_eap, validate_model_for_eap
from model_audit_matrix import ModelAuditCase, UNSUPPORTED_BRIDGE_FORK
from conftest import hf_or_skip


HOOKED_LOAD_KWARGS = {
    "fold_ln": False,
    "center_writing_weights": False,
    "center_unembed": False,
    "fold_value_biases": False,
}
BRIDGE_COMPAT_KWARGS = {"no_processing": True}


def require_enabled(env_var: str, message: str) -> None:
    if os.environ.get(env_var) != "1":
        pytest.skip(message)


def require_case_access(case: ModelAuditCase) -> None:
    if case.requires_hf_token and not _hf_token():
        pytest.skip(f"{case.checkpoint} requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN.")


def resolve_device(case: ModelAuditCase, default: str = "cpu") -> str:
    requested = os.environ.get("EAP_MODEL_AUDIT_DEVICE", default).strip().lower()
    if requested == "gpu":
        requested = "cuda"

    if case.requires_gpu:
        requested = "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"{case.case_id} requires CUDA but CUDA is unavailable.")
    if not requested:
        requested = default
    return requested


def tiny_dataloader():
    return [(["The cat sat on the mat"], ["The dog sat on the mat"], torch.tensor([0]))]


def metric(logits, clean_logits, input_lengths, label):
    batch = torch.arange(logits.size(0), device=logits.device)
    final_pos = input_lengths.to(logits.device) - 1
    return logits[batch, final_pos, 0].sum()


def load_bridge_model(case: ModelAuditCase, device: str) -> Any:
    if UNSUPPORTED_BRIDGE_FORK in case.expected_caveats:
        pytest.skip(
            f"{case.case_id} is intentionally classified as unsupported in TransformerBridge: "
            "its attention layers do not expose the pre-Q/K/V hook fork EAP requires."
        )
    pytest.importorskip("transformer_lens")
    try:
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError:
        pytest.skip("TransformerBridge is unavailable in this TransformerLens install.")

    bridge = hf_or_skip(
        f"{case.case_id} TransformerBridge",
        TransformerBridge.boot_transformers,
        case.checkpoint,
        device=device,
    )
    model = prepare_model_for_eap(bridge, compatibility_mode_kwargs=BRIDGE_COMPAT_KWARGS)
    model.eval()
    return model


def load_hooked_model(case: ModelAuditCase, device: str) -> Any:
    transformer_lens = pytest.importorskip("transformer_lens")
    hooked_cls = getattr(transformer_lens, "HookedTransformer", None)
    if hooked_cls is None:
        pytest.skip("HookedTransformer is unavailable in this TransformerLens install.")

    model = hf_or_skip(
        f"{case.case_id} HookedTransformer",
        hooked_cls.from_pretrained,
        case.checkpoint,
        device=device,
        **HOOKED_LOAD_KWARGS,
    )
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    n_heads = getattr(model.cfg, "n_heads", None)
    n_key_value_heads = getattr(model.cfg, "n_key_value_heads", None)
    if (
        n_heads is not None
        and n_key_value_heads is not None
        and n_key_value_heads != n_heads
    ):
        model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    return model


def run_smoke_attribute_and_baseline(model: Any) -> Graph:
    graph = Graph.from_model(model)
    attribute(model, graph, tiny_dataloader(), metric, method="EAP", quiet=True)
    attribute(
        model,
        graph,
        tiny_dataloader(),
        metric,
        method="EAP-IG-inputs",
        ig_steps=2,
        quiet=True,
    )
    baseline = evaluate_baseline(model, tiny_dataloader(), [metric], quiet=True)
    assert torch.isfinite(baseline).all()
    assert torch.isfinite(graph.scores).all()
    return graph


def validate_prepared_model(model: Any, case: ModelAuditCase | None = None) -> None:
    validate_model_for_eap(model)


def _hf_token() -> str:
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token
    return ""


def release_models(*models: Any) -> None:
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
