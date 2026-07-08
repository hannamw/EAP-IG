from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Sequence


LoadMode = Literal["bridge", "both"]
UNSUPPORTED_BRIDGE_FORK = "bridge_attention_fork_unavailable"


@dataclass(frozen=True)
class ModelAuditCase:
    case_id: str
    family: str
    checkpoint: str
    load_mode: LoadMode
    requires_hf_token: bool = False
    requires_gpu: bool = False
    expected_caveats: tuple[str, ...] = ()


SMOKE_CASES: tuple[ModelAuditCase, ...] = (
    ModelAuditCase(
        case_id="gpt2-smoke",
        family="gpt2",
        checkpoint="gpt2",
        load_mode="bridge",
    ),
    ModelAuditCase(
        case_id="llama-smoke",
        family="llama",
        checkpoint="trl-internal-testing/tiny-random-LlamaForCausalLM",
        load_mode="bridge",
    ),
    ModelAuditCase(
        case_id="qwen-smoke",
        family="qwen",
        checkpoint="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        load_mode="bridge",
        expected_caveats=("gqa",),
    ),
    ModelAuditCase(
        case_id="mistral-smoke",
        family="mistral",
        checkpoint="trl-internal-testing/tiny-MistralForCausalLM-0.1",
        load_mode="bridge",
        expected_caveats=("gqa", UNSUPPORTED_BRIDGE_FORK),
    ),
    ModelAuditCase(
        case_id="gemma-smoke",
        family="gemma",
        checkpoint="trl-internal-testing/tiny-Gemma2ForCausalLM",
        load_mode="bridge",
        expected_caveats=("gqa",),
    ),
)


CERTIFICATION_CASES: tuple[ModelAuditCase, ...] = (
    ModelAuditCase(
        case_id="gpt2-cert",
        family="gpt2",
        checkpoint="gpt2",
        load_mode="both",
        requires_gpu=True,
    ),
    ModelAuditCase(
        case_id="llama-cert",
        family="llama",
        checkpoint="meta-llama/Llama-3.2-3B",
        load_mode="both",
        requires_hf_token=True,
        requires_gpu=True,
        expected_caveats=("gqa",),
    ),
    ModelAuditCase(
        case_id="qwen-cert",
        family="qwen",
        checkpoint="Qwen/Qwen3-0.6B",
        load_mode="both",
        requires_gpu=True,
        expected_caveats=("gqa",),
    ),
    ModelAuditCase(
        case_id="mistral-cert",
        family="mistral",
        checkpoint="mistralai/Mistral-7B-v0.1",
        load_mode="both",
        requires_hf_token=True,
        requires_gpu=True,
        expected_caveats=("gqa", UNSUPPORTED_BRIDGE_FORK),
    ),
    ModelAuditCase(
        case_id="gemma-cert",
        family="gemma",
        checkpoint="google/gemma-2-2b",
        load_mode="both",
        requires_hf_token=True,
        requires_gpu=True,
        expected_caveats=("gqa",),
    ),
    ModelAuditCase(
        case_id="ministral-cert",
        family="ministral",
        checkpoint="mistralai/Ministral-8B-Instruct-2410",
        load_mode="bridge",
        requires_hf_token=True,
        requires_gpu=True,
        expected_caveats=("gqa", UNSUPPORTED_BRIDGE_FORK),
    ),
)


def select_cases(cases: Sequence[ModelAuditCase]) -> tuple[ModelAuditCase, ...]:
    requested = os.environ.get("EAP_MODEL_AUDIT_FAMILIES", "").strip()
    if not requested:
        return tuple(cases)

    wanted = {token.strip().lower() for token in requested.split(",") if token.strip()}
    selected = []
    for case in cases:
        if case.family.lower() in wanted or case.case_id.lower() in wanted:
            selected.append(case)
    return tuple(selected)
