from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch


_EAP_PREPARED_FLAG = "_eap_prepared_for_compatibility"
_EAP_COMPAT_FLAG = "_eap_enabled_bridge_compatibility"
_EAP_UNSUPPORTED_FEATURES = "_eap_unsupported_bridge_features"
_EAP_MATERIALIZED_GQA_FLAG = "_eap_materialized_ungrouped_gqa"
_EAP_ORIGINAL_N_KEY_VALUE_HEADS = "_eap_original_n_key_value_heads"
_MISSING = object()


@runtime_checkable
class EAPModelProtocol(Protocol):
    cfg: Any
    tokenizer: Any

    def to_tokens(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def hooks(self, *args: Any, **kwargs: Any) -> Any: ...

    def run_with_hooks(self, *args: Any, **kwargs: Any) -> Any: ...

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class EAPModelAdapter:
    """A thin delegating wrapper over HookedTransformer-like models."""

    def __init__(self, model: EAPModelProtocol):
        self.model = model

    @property
    def cfg(self) -> Any:
        return self.model.cfg

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    def to_tokens(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model.to_tokens(*args, **kwargs)

    def hooks(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.hooks(*args, **kwargs)

    def run_with_hooks(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.run_with_hooks(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.zero_grad(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def cfg_get(cfg: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(cfg, dict):
        if name in cfg:
            return cfg[name]
    elif hasattr(cfg, name):
        return getattr(cfg, name)
    else:
        try:
            return cfg[name]
        except (KeyError, TypeError):
            pass

    if default is _MISSING:
        raise AttributeError(f"Config is missing required field {name!r}")
    return default


def cfg_set(cfg: Any, name: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[name] = value
    else:
        setattr(cfg, name, value)


def is_bridge_like(model: Any) -> bool:
    return (
        hasattr(model, "enable_compatibility_mode")
        and hasattr(model, "run_with_hooks")
        and hasattr(model, "hook_aliases")
    )


def _unsupported_bridge_features(model: Any) -> dict[str, str]:
    unsupported = getattr(model, _EAP_UNSUPPORTED_FEATURES, None)
    if unsupported is None:
        unsupported = {}
        setattr(model, _EAP_UNSUPPORTED_FEATURES, unsupported)
    return unsupported


def _feature_name_from_setter(method_name: str) -> str:
    return method_name[len("set_") :] if method_name.startswith("set_") else method_name


def _feature_is_unsupported(model: Any, feature_name: str) -> bool:
    return feature_name in _unsupported_bridge_features(model)


def _bridge_compatibility_enabled(model: Any) -> bool:
    return bool(
        getattr(model, _EAP_COMPAT_FLAG, False)
        or getattr(model, "compatibility_mode_enabled", False)
        or getattr(model, "_compatibility_mode_enabled", False)
    )


def _bridge_has_legacy_eap_hook_semantics(model: Any) -> bool:
    return bool(
        hasattr(model, "set_use_hook_mlp_in")
        and hasattr(model, "set_use_split_qkv_input")
    )


def _call_if_present(model: Any, method_name: str, *args: Any, **kwargs: Any) -> bool:
    method = getattr(model, method_name, None)
    if method is None:
        return False
    try:
        method(*args, **kwargs)
    except NotImplementedError as exc:
        _unsupported_bridge_features(model)[_feature_name_from_setter(method_name)] = str(exc)
        return False
    return True


def _repeat_linear_output_heads(
    linear: torch.nn.Linear,
    *,
    n_heads: int,
    n_key_value_heads: int,
    d_head: int,
) -> torch.nn.Linear:
    expected_output_features = n_key_value_heads * d_head
    if linear.out_features == n_heads * d_head:
        return linear
    if linear.out_features != expected_output_features:
        raise NotImplementedError(
            "K/V projection output width does not match n_key_value_heads * d_head"
        )

    repeat = n_heads // n_key_value_heads
    replacement = torch.nn.Linear(
        linear.in_features,
        n_heads * d_head,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    replacement.training = linear.training
    replacement.weight.requires_grad = linear.weight.requires_grad
    with torch.no_grad():
        weight = linear.weight.reshape(n_key_value_heads, d_head, linear.in_features)
        replacement.weight.copy_(
            weight.repeat_interleave(repeat, dim=0).reshape(n_heads * d_head, linear.in_features)
        )
        if linear.bias is not None and replacement.bias is not None:
            replacement.bias.requires_grad = linear.bias.requires_grad
            bias = linear.bias.reshape(n_key_value_heads, d_head)
            replacement.bias.copy_(bias.repeat_interleave(repeat, dim=0).reshape(n_heads * d_head))
    return replacement


def _materialize_bridge_gqa_ungroup(
    model: Any,
    cfg: Any,
    *,
    n_heads: int,
    n_key_value_heads: int,
) -> None:
    if getattr(model, _EAP_MATERIALIZED_GQA_FLAG, False):
        return
    if n_heads % n_key_value_heads != 0:
        _unsupported_bridge_features(model)["ungroup_grouped_query_attention"] = (
            "n_heads must be divisible by n_key_value_heads to materialize ungrouped GQA"
        )
        return

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return

    d_head = cfg_get(cfg, "d_head", None)
    if d_head is None:
        d_model = cfg_get(cfg, "d_model", None)
        if d_model is None or d_model % n_heads != 0:
            _unsupported_bridge_features(model)["ungroup_grouped_query_attention"] = (
                "Cannot infer d_head needed to materialize ungrouped GQA"
            )
            return
        d_head = d_model // n_heads
    d_head = int(d_head)

    materialized_any = False
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue

        for projection_name in ("k", "v"):
            projection = getattr(attn, projection_name, None)
            original = getattr(projection, "original_component", None)
            if original is None or not isinstance(original, torch.nn.Linear):
                _unsupported_bridge_features(model)["ungroup_grouped_query_attention"] = (
                    "Bridge K/V projection is not a materializable torch.nn.Linear"
                )
                return
            projection.set_original_component(
                _repeat_linear_output_heads(
                    original,
                    n_heads=n_heads,
                    n_key_value_heads=n_key_value_heads,
                    d_head=d_head,
                )
            )
            materialized_any = True

        hf_attn = getattr(attn, "original_component", None)
        attn_config = getattr(attn, "config", None)
        if attn_config is not None:
            cfg_set(attn_config, "n_key_value_heads", n_heads)
        if hf_attn is not None:
            if hasattr(hf_attn, "num_key_value_groups"):
                setattr(hf_attn, "num_key_value_groups", 1)
            for attr_name in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
                if hasattr(hf_attn, attr_name):
                    setattr(hf_attn, attr_name, n_heads)
            hf_config = getattr(hf_attn, "config", None)
            if hf_config is not None and hasattr(hf_config, "num_key_value_heads"):
                setattr(hf_config, "num_key_value_heads", n_heads)

    if materialized_any:
        setattr(model, _EAP_MATERIALIZED_GQA_FLAG, True)
        setattr(model, _EAP_ORIGINAL_N_KEY_VALUE_HEADS, n_key_value_heads)
        cfg_set(cfg, "n_key_value_heads", n_heads)


def _configure_bridge_hooks(model: Any) -> None:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and cfg_get(cfg, "use_attn_in", False):
        _call_if_present(model, "set_use_attn_in", False)
        if cfg_get(cfg, "use_attn_in", False):
            cfg_set(cfg, "use_attn_in", False)

    if (
        not _call_if_present(model, "set_use_attn_result", True)
        and cfg is not None
        and not _feature_is_unsupported(model, "use_attn_result")
    ):
        cfg_set(cfg, "use_attn_result", True)

    if (
        not _call_if_present(model, "set_use_split_qkv_input", True)
        and cfg is not None
        and not _feature_is_unsupported(model, "use_split_qkv_input")
    ):
        cfg_set(cfg, "use_split_qkv_input", True)

    if (
        not _call_if_present(model, "set_use_hook_mlp_in", True)
        and cfg is not None
        and not _feature_is_unsupported(model, "use_hook_mlp_in")
    ):
        cfg_set(cfg, "use_hook_mlp_in", True)

    if cfg is not None:
        n_heads = cfg_get(cfg, "n_heads", None)
        n_key_value_heads = cfg_get(cfg, "n_key_value_heads", None)
        if (
            n_heads is not None
            and n_key_value_heads is not None
            and n_key_value_heads != n_heads
        ):
            if (
                not cfg_get(cfg, "ungroup_grouped_query_attention", False)
                and not _call_if_present(model, "set_ungroup_grouped_query_attention", True)
            ):
                cfg_set(cfg, "ungroup_grouped_query_attention", True)
            _materialize_bridge_gqa_ungroup(
                model,
                cfg,
                n_heads=int(n_heads),
                n_key_value_heads=int(n_key_value_heads),
            )


def prepare_model_for_eap(
    model: EAPModelProtocol | EAPModelAdapter,
    *,
    auto_enable_bridge_compat: bool = True,
    compatibility_mode_kwargs: Optional[dict[str, Any]] = None,
) -> EAPModelAdapter:
    """Prepare a HookedTransformer-like model for EAP and return a delegating adapter."""

    if isinstance(model, EAPModelAdapter):
        original_model = model.model
        adapter = model
    else:
        original_model = model
        adapter = EAPModelAdapter(model)

    if is_bridge_like(original_model):
        if auto_enable_bridge_compat and not _bridge_compatibility_enabled(original_model):
            original_model.enable_compatibility_mode(**(compatibility_mode_kwargs or {}))
            setattr(original_model, _EAP_COMPAT_FLAG, True)

        if not getattr(original_model, _EAP_PREPARED_FLAG, False):
            _configure_bridge_hooks(original_model)
            setattr(original_model, _EAP_PREPARED_FLAG, True)

    return adapter


def get_model_device(model: Any) -> torch.device | str:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        device = cfg_get(cfg, "device", None)
        if device is not None:
            return device

    parameters = getattr(model, "parameters", None)
    if parameters is not None:
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            pass

    return torch.device("cpu")


def get_model_dtype(model: Any) -> torch.dtype:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        dtype = cfg_get(cfg, "dtype", None)
        if dtype is not None:
            return dtype
    return torch.float32


def get_score_accumulation_dtype(model: Any) -> torch.dtype:
    dtype = get_model_dtype(model)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def validate_model_for_eap(model: Any) -> None:
    cfg = model.cfg
    unsupported = getattr(model, _EAP_UNSUPPORTED_FEATURES, None) or {}
    if unsupported:
        details = "; ".join(f"{name}: {message}" for name, message in unsupported.items())
        raise NotImplementedError(
            "Model bridge does not expose the hook surface required for EAP. "
            f"{details}"
        )

    required_flags = [
        ("use_attn_result", "Model must be configured to use attention result"),
        ("use_split_qkv_input", "Model must be configured to use split qkv inputs"),
        ("use_hook_mlp_in", "Model must be configured to use hook MLP in"),
    ]
    missing = [
        f"{message} (model.cfg.{flag})"
        for flag, message in required_flags
        if hasattr(cfg, flag) and not cfg_get(cfg, flag)
    ]

    if is_bridge_like(model) and not _bridge_has_legacy_eap_hook_semantics(model):
        missing.append(
            "TransformerBridge attribution/evaluation requires a TransformerLens "
            "build with legacy-compatible EAP hook semantics; upgrade "
            "transformer-lens to >=3.5.1"
        )

    n_heads = cfg_get(cfg, "n_heads", None)
    n_key_value_heads = cfg_get(cfg, "n_key_value_heads", None)
    if n_key_value_heads is not None and n_key_value_heads != n_heads:
        if not cfg_get(cfg, "ungroup_grouped_query_attention", False):
            missing.append(
                "Model must ungroup grouped query attention "
                "(model.cfg.ungroup_grouped_query_attention)"
            )

    if missing:
        raise AssertionError("; ".join(missing))
