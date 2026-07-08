import math

import pytest
import torch

from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from eap.utils import _maybe_expand_grouped_query_tensor, tokenize_plus
from model_audit_helpers import (
    load_bridge_model,
    metric,
    require_case_access,
    require_enabled,
    resolve_device,
    validate_prepared_model,
)
from model_audit_matrix import SMOKE_CASES, select_cases


pytestmark = [
    pytest.mark.integration,
    pytest.mark.model_audit_math,
]


MATH_CASES = tuple(case for case in SMOKE_CASES if case.family in {"gpt2", "llama", "qwen", "gemma", "mistral"})
GQA_CASES = tuple(case for case in MATH_CASES if "gqa" in case.expected_caveats)
FAITHFULNESS_CASES = MATH_CASES


def _single_batch(clean: str = "The cat sat on the mat", corrupted: str = "The dog sat on the mat"):
    return [([clean], [corrupted], torch.tensor([0]))]


def _final_logit_metric(model, texts):
    tokens, attention_mask, input_lengths, _ = tokenize_plus(model, texts)
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    input_lengths = input_lengths.to(device)
    with torch.inference_mode():
        logits = model(tokens, attention_mask=attention_mask)
    return metric(logits, None, input_lengths, torch.tensor([0], device=device)).item()


def _real_edge_values_for_child(graph: Graph, child_name: str) -> torch.Tensor:
    values = []
    for edge in graph.edges.values():
        if edge.child.name == child_name:
            values.append(graph.scores[edge.matrix_index].detach().float().cpu())
    if not values:
        return torch.empty(0)
    return torch.stack(values)


def _spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel():
        raise ValueError("Inputs must have the same number of elements.")
    if x.numel() < 2:
        raise ValueError("Need at least two elements for rank correlation.")

    x_ranks = torch.argsort(torch.argsort(x)).float()
    y_ranks = torch.argsort(torch.argsort(y)).float()
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if denom.item() == 0:
        return float("nan")
    return torch.dot(x_centered, y_centered).div(denom).item()


def _sign_agreement(x: torch.Tensor, y: torch.Tensor) -> float:
    return x.sign().eq(y.sign()).float().mean().item()


@pytest.mark.parametrize("case", select_cases(MATH_CASES), ids=lambda case: case.case_id)
def test_completeness_axiom(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_CERT",
        "Set EAP_RUN_MODEL_AUDIT_CERT=1 to run model-audit mathematical validation tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cuda")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)

    dataloader = _single_batch()
    graph = Graph.from_model(model)
    attribute(
        model,
        graph,
        dataloader,
        metric,
        method="EAP-IG-inputs",
        ig_steps=20,
        quiet=True,
    )

    clean_score = _final_logit_metric(model, dataloader[0][0])
    corrupted_score = _final_logit_metric(model, dataloader[0][1])
    delta = corrupted_score - clean_score

    logit_edge_scores = _real_edge_values_for_child(graph, "logits")
    assert logit_edge_scores.numel() > 0, "No parent -> logits edges were collected."
    assert math.isfinite(delta)
    assert torch.isfinite(logit_edge_scores).all()
    assert math.isclose(logit_edge_scores.sum().item(), delta, rel_tol=0.25, abs_tol=0.25)


@pytest.mark.parametrize("case", select_cases(MATH_CASES), ids=lambda case: case.case_id)
def test_null_edge_leakage(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_CERT",
        "Set EAP_RUN_MODEL_AUDIT_CERT=1 to run model-audit mathematical validation tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cuda")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)

    dataloader = _single_batch(clean="Null path prompt", corrupted="Null path prompt")
    graph = Graph.from_model(model)
    attribute(
        model,
        graph,
        dataloader,
        metric,
        method="EAP-IG-inputs",
        ig_steps=4,
        quiet=True,
    )

    assert torch.count_nonzero(graph.scores).item() == 0
    assert torch.count_nonzero(graph.scores[~graph.real_edge_mask]).item() == 0


@pytest.mark.parametrize("case", select_cases(GQA_CASES), ids=lambda case: case.case_id)
def test_gqa_gradient_broadcast(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_SMOKE",
        "Set EAP_RUN_MODEL_AUDIT_SMOKE=1 to run GQA mathematical validation tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cpu")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)

    tokens, attention_mask, input_lengths, _ = tokenize_plus(model, ["Grouped query attention"])
    model_device = next(model.parameters()).device
    tokens = tokens.to(model_device)
    attention_mask = attention_mask.to(model_device)

    grads = {}

    def capture(name):
        def hook_fn(grad, hook):
            grads[name] = grad.detach().cpu()

        return hook_fn

    model.zero_grad()
    with model.hooks(
        bwd_hooks=[
            ("blocks.0.hook_k_input", capture("k")),
            ("blocks.0.hook_v_input", capture("v")),
        ]
    ):
        logits = model(tokens, attention_mask=attention_mask)
        batch = torch.arange(logits.size(0), device=logits.device)
        final_pos = input_lengths.to(logits.device) - 1
        logits[batch, final_pos, 0].sum().backward()

    n_heads = model.cfg.n_heads
    expanded_k = _maybe_expand_grouped_query_tensor(model, grads["k"])
    expanded_v = _maybe_expand_grouped_query_tensor(model, grads["v"])

    assert grads.keys() == {"k", "v"}
    assert grads["k"].ndim == 4 and grads["v"].ndim == 4
    assert expanded_k.shape[2] == n_heads
    assert expanded_v.shape[2] == n_heads
    assert torch.isfinite(expanded_k).all()
    assert torch.isfinite(expanded_v).all()

    graph = Graph.from_model(model)
    attribute(model, graph, _single_batch(), metric, method="EAP", quiet=True)
    assert torch.isfinite(graph.scores).all()


@pytest.mark.parametrize("case", select_cases(GQA_CASES), ids=lambda case: case.case_id)
def test_gqa_bridge_kv_inputs_are_ungrouped_for_eap(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_SMOKE",
        "Set EAP_RUN_MODEL_AUDIT_SMOKE=1 to run GQA hook ungrouping validation tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cpu")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)

    assert model.cfg.ungroup_grouped_query_attention is True
    assert model.cfg.n_key_value_heads == model.cfg.n_heads

    tokens, attention_mask, _, _ = tokenize_plus(model, ["Grouped query attention"])
    model_device = next(model.parameters()).device
    tokens = tokens.to(model_device)
    attention_mask = attention_mask.to(model_device)

    shapes = {}

    def capture(name):
        def hook_fn(activations, hook):
            shapes[name] = tuple(activations.shape)

        return hook_fn

    with model.hooks(
        fwd_hooks=[
            ("blocks.0.hook_k_input", capture("k")),
            ("blocks.0.hook_v_input", capture("v")),
        ]
    ):
        model(tokens, attention_mask=attention_mask)

    assert shapes["k"][2] == model.cfg.n_heads
    assert shapes["v"][2] == model.cfg.n_heads


@pytest.mark.parametrize("case", select_cases(MATH_CASES), ids=lambda case: case.case_id)
def test_bfloat16_accumulation(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_SMOKE",
        "Set EAP_RUN_MODEL_AUDIT_SMOKE=1 to run mathematical precision validation tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cpu")

    model = load_bridge_model(case, device=device)
    model = model.to(torch.bfloat16)
    validate_prepared_model(model, case)

    graph = Graph.from_model(model)
    attribute(
        model,
        graph,
        _single_batch(),
        metric,
        method="EAP-IG-inputs",
        ig_steps=8,
        quiet=True,
    )

    assert graph.scores.dtype == torch.float32
    assert torch.isfinite(graph.scores).all()


@pytest.mark.slow
@pytest.mark.parametrize("case", select_cases(FAITHFULNESS_CASES), ids=lambda case: case.case_id)
def test_exact_patching_faithfulness(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_CERT",
        "Set EAP_RUN_MODEL_AUDIT_CERT=1 to run model-audit mathematical validation tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cuda")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)

    dataloader = _single_batch()
    ig_graph = Graph.from_model(model)
    attribute(
        model,
        ig_graph,
        dataloader,
        metric,
        method="EAP-IG-inputs",
        ig_steps=12,
        quiet=True,
    )
    candidate_edges = []
    for edge_name, edge in ig_graph.edges.items():
        if edge.child.name != "logits":
            continue
        ig_value = ig_graph.scores[edge.matrix_index].item()
        if math.isfinite(ig_value):
            candidate_edges.append((abs(ig_value), edge_name, ig_value))

    assert len(candidate_edges) >= 8, "Not enough finite real-edge scores for the faithfulness check."
    candidate_edges.sort(reverse=True)
    sampled_edges = candidate_edges[:20]

    baseline = evaluate_baseline(model, dataloader, [metric], quiet=True).mean().item()
    exact_scores = []
    ig_scores = []
    for _, edge_name, ig_value in sampled_edges:
        exact_graph = Graph.from_model(model)
        exact_graph.in_graph |= exact_graph.real_edge_mask
        exact_graph.edges[edge_name].in_graph = False
        exact_value = (
            evaluate_graph(model, exact_graph, dataloader, metric, quiet=True, skip_clean=True).mean().item()
            - baseline
        )
        if math.isfinite(exact_value):
            exact_scores.append(exact_value)
            ig_scores.append(ig_value)

    assert len(ig_scores) >= 8, "Not enough exact-patching scores were computed for the faithfulness check."
    exact_scores = torch.tensor(exact_scores)
    ig_scores = torch.tensor(ig_scores)
    correlation = _spearman_rank_correlation(exact_scores.abs(), ig_scores.abs())
    agreement = _sign_agreement(exact_scores, ig_scores)
    assert correlation > 0.8, f"Spearman correlation between exact patching and EAP-IG magnitudes is too low: {correlation}"
    assert agreement > 0.8, f"Exact patching and EAP-IG disagree on edge direction too often: {agreement}"
