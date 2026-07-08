import pytest
import torch

from eap.attribute import attribute
from eap.graph import Graph
from eap.utils import tokenize_plus
from model_audit_helpers import (
    load_bridge_model,
    load_hooked_model,
    metric,
    require_case_access,
    require_enabled,
    release_models,
    resolve_device,
    run_smoke_attribute_and_baseline,
    tiny_dataloader,
    validate_prepared_model,
)
from model_audit_matrix import CERTIFICATION_CASES, select_cases


pytestmark = [
    pytest.mark.integration,
    pytest.mark.model_audit_certification,
]


def _cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() == 0:
        return 1.0
    return torch.dot(x, y).div(denom).item()


@pytest.mark.parametrize("case", select_cases(CERTIFICATION_CASES), ids=lambda case: case.case_id)
def test_bridge_model_audit_certification(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_CERT",
        "Set EAP_RUN_MODEL_AUDIT_CERT=1 to run model-audit certification tests.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cuda")

    model = load_bridge_model(case, device=device)
    validate_prepared_model(model, case)

    first_graph = run_smoke_attribute_and_baseline(model)
    second_graph = run_smoke_attribute_and_baseline(model)
    assert first_graph.scores.shape == second_graph.scores.shape
    assert torch.isfinite(first_graph.scores).all()
    assert torch.isfinite(second_graph.scores).all()


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in select_cases(CERTIFICATION_CASES)
        if case.load_mode == "both" and "gqa" not in case.expected_caveats
    ],
    ids=lambda case: case.case_id,
)
def test_hooked_and_bridge_parity_on_certification_models(case):
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_CERT",
        "Set EAP_RUN_MODEL_AUDIT_CERT=1 to run model-audit certification tests.",
    )
    require_enabled(
        "EAP_RUN_MODEL_AUDIT_PARITY",
        "Set EAP_RUN_MODEL_AUDIT_PARITY=1 to run certification parity checks.",
    )
    require_case_access(case)
    device = resolve_device(case, default="cuda")

    hooked = load_hooked_model(case, device=device)

    clean = ["The cat sat on the mat"]
    hooked_graph = Graph.from_model(hooked)
    hooked_tokens, hooked_mask, hooked_lengths, _ = tokenize_plus(hooked, clean)
    with torch.inference_mode():
        hooked_logits = hooked(hooked_tokens, attention_mask=hooked_mask).detach().cpu()
    attribute(hooked, hooked_graph, tiny_dataloader(), metric, method="EAP", quiet=True)
    hooked_scores = hooked_graph.scores.detach().cpu()
    hooked_real_edge_mask = hooked_graph.real_edge_mask.detach().cpu()
    hooked_tokens = hooked_tokens.detach().cpu()
    hooked_mask = hooked_mask.detach().cpu()
    hooked_lengths = hooked_lengths.detach().cpu()
    release_models(hooked, hooked_graph)

    bridge = load_bridge_model(case, device=device)
    validate_prepared_model(bridge, case)
    bridge_tokens, bridge_mask, bridge_lengths, _ = tokenize_plus(bridge, clean)

    torch.testing.assert_close(bridge_tokens.cpu(), hooked_tokens)
    torch.testing.assert_close(bridge_mask.cpu(), hooked_mask)
    torch.testing.assert_close(bridge_lengths.cpu(), hooked_lengths)

    with torch.inference_mode():
        bridge_logits = bridge(bridge_tokens, attention_mask=bridge_mask)
    torch.testing.assert_close(bridge_logits.cpu(), hooked_logits, rtol=1e-2, atol=1e-2)

    bridge_graph = Graph.from_model(bridge)
    attribute(bridge, bridge_graph, tiny_dataloader(), metric, method="EAP", quiet=True)
    bridge_mask = bridge_graph.real_edge_mask.cpu()
    torch.testing.assert_close(bridge_mask, hooked_real_edge_mask)

    bridge_scores = bridge_graph.scores.cpu()[bridge_mask]
    hooked_scores = hooked_scores[hooked_real_edge_mask]
    torch.testing.assert_close(bridge_scores.sum(), hooked_scores.sum(), rtol=5e-2, atol=5e-2)
    assert _cosine_similarity(bridge_scores, hooked_scores) > 0.995
