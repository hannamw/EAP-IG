from types import SimpleNamespace

import pytest
import torch

from eap.model_adapter import EAPModelAdapter, prepare_model_for_eap, validate_model_for_eap


class HookContext:
    def __enter__(self):
        return "entered"

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeBridge:
    def __init__(self, *, n_key_value_heads=None, ungroup_grouped_query_attention=False):
        self.cfg = SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            use_attn_in=True,
            use_attn_result=False,
            use_split_qkv_input=False,
            use_hook_mlp_in=False,
            n_heads=2,
            n_key_value_heads=n_key_value_heads,
            ungroup_grouped_query_attention=ungroup_grouped_query_attention,
        )
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.hook_aliases = {}
        self.compatibility_calls = 0
        self.attn_in_calls = []
        self.attn_result_calls = []
        self.split_qkv_calls = []
        self.zero_grad_calls = 0

    def enable_compatibility_mode(self, **kwargs):
        self.compatibility_calls += 1
        self.compatibility_kwargs = kwargs

    def set_use_attn_in(self, value):
        self.attn_in_calls.append(value)
        self.cfg.use_attn_in = value

    def set_use_attn_result(self, value):
        self.attn_result_calls.append(value)
        self.cfg.use_attn_result = value

    def set_use_split_qkv_input(self, value):
        self.split_qkv_calls.append(value)
        self.cfg.use_split_qkv_input = value

    def run_with_hooks(self, *args, **kwargs):
        return args, kwargs

    def hooks(self, *args, **kwargs):
        return HookContext()

    def to_tokens(self, inputs, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def zero_grad(self):
        self.zero_grad_calls += 1

    def __call__(self, tokens, **kwargs):
        return tokens + 1


class CompatibleFakeBridge(FakeBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hook_mlp_in_calls = []

    def set_use_hook_mlp_in(self, value):
        self.hook_mlp_in_calls.append(value)
        self.cfg.use_hook_mlp_in = value


def test_prepare_bridge_enables_compatibility_and_required_hooks_once():
    bridge = CompatibleFakeBridge()

    adapter = prepare_model_for_eap(
        bridge,
        compatibility_mode_kwargs={"no_processing": True},
    )
    prepare_model_for_eap(adapter)

    assert isinstance(adapter, EAPModelAdapter)
    assert bridge.compatibility_calls == 1
    assert bridge.compatibility_kwargs == {"no_processing": True}
    assert bridge.attn_in_calls == [False]
    assert bridge.attn_result_calls == [True]
    assert bridge.split_qkv_calls == [True]
    assert bridge.hook_mlp_in_calls == [True]
    assert bridge.cfg.use_attn_in is False
    assert bridge.cfg.use_attn_result is True
    assert bridge.cfg.use_split_qkv_input is True
    assert bridge.cfg.use_hook_mlp_in is True


def test_adapter_delegates_model_surface():
    bridge = FakeBridge()
    adapter = prepare_model_for_eap(bridge)

    assert adapter.to_tokens(["hello"]).tolist() == [[1, 2, 3]]
    with adapter.hooks() as value:
        assert value == "entered"
    assert adapter(torch.tensor([1])).tolist() == [2]
    adapter.zero_grad()
    assert bridge.zero_grad_calls == 1


def test_validate_accepts_prepared_bridge_by_default():
    bridge = CompatibleFakeBridge()
    adapter = prepare_model_for_eap(bridge)

    validate_model_for_eap(adapter)


def test_validate_rejects_bridge_without_legacy_hook_capability_by_default():
    bridge = FakeBridge()
    adapter = prepare_model_for_eap(bridge)

    with pytest.raises(AssertionError, match="legacy-compatible EAP hook semantics"):
        validate_model_for_eap(adapter)


def test_prepare_bridge_auto_ungroups_grouped_query_attention():
    bridge = CompatibleFakeBridge(n_key_value_heads=1, ungroup_grouped_query_attention=False)

    adapter = prepare_model_for_eap(bridge)

    assert isinstance(adapter, EAPModelAdapter)
    assert bridge.cfg.ungroup_grouped_query_attention is True
    validate_model_for_eap(adapter)


class FakeProjection:
    def __init__(self, original_component):
        self._original_component = original_component

    @property
    def original_component(self):
        return self._original_component

    def set_original_component(self, original_component):
        self._original_component = original_component


class MaterializableGQABridge(CompatibleFakeBridge):
    def __init__(self):
        super().__init__(n_key_value_heads=2, ungroup_grouped_query_attention=False)
        self.cfg.n_heads = 4
        self.cfg.d_head = 3
        self.cfg.d_model = 12

        k = torch.nn.Linear(12, 6, bias=True)
        v = torch.nn.Linear(12, 6, bias=False)
        with torch.no_grad():
            k.weight.copy_(torch.arange(72, dtype=torch.float32).reshape(6, 12))
            k.bias.copy_(torch.arange(6, dtype=torch.float32))
            v.weight.copy_(torch.arange(72, 144, dtype=torch.float32).reshape(6, 12))

        self.blocks = [
            SimpleNamespace(
                attn=SimpleNamespace(
                    k=FakeProjection(k),
                    v=FakeProjection(v),
                    config=SimpleNamespace(n_key_value_heads=2),
                    original_component=SimpleNamespace(num_key_value_groups=2),
                )
            )
        ]


def test_prepare_bridge_materializes_grouped_query_attention():
    bridge = MaterializableGQABridge()
    original_k = bridge.blocks[0].attn.k.original_component
    original_v = bridge.blocks[0].attn.v.original_component

    adapter = prepare_model_for_eap(bridge)
    prepare_model_for_eap(adapter)

    k = bridge.blocks[0].attn.k.original_component
    v = bridge.blocks[0].attn.v.original_component
    assert bridge.cfg.ungroup_grouped_query_attention is True
    assert bridge.cfg.n_key_value_heads == bridge.cfg.n_heads
    assert bridge.blocks[0].attn.config.n_key_value_heads == bridge.cfg.n_heads
    assert bridge.blocks[0].attn.original_component.num_key_value_groups == 1
    assert k.out_features == bridge.cfg.n_heads * bridge.cfg.d_head
    assert v.out_features == bridge.cfg.n_heads * bridge.cfg.d_head
    torch.testing.assert_close(
        k.weight.reshape(4, 3, 12),
        original_k.weight.reshape(2, 3, 12).repeat_interleave(2, dim=0),
    )
    torch.testing.assert_close(
        k.bias.reshape(4, 3),
        original_k.bias.reshape(2, 3).repeat_interleave(2, dim=0),
    )
    torch.testing.assert_close(
        v.weight.reshape(4, 3, 12),
        original_v.weight.reshape(2, 3, 12).repeat_interleave(2, dim=0),
    )
    validate_model_for_eap(adapter)


class FakeUnsupportedAttentionBridge(CompatibleFakeBridge):
    def set_use_attn_result(self, value):
        raise NotImplementedError("use_attn_result: unsupported attention fork")

    def set_use_split_qkv_input(self, value):
        raise NotImplementedError("use_split_qkv_input: unsupported attention fork")


def test_validate_rejects_bridge_without_required_attention_fork():
    bridge = FakeUnsupportedAttentionBridge()

    adapter = prepare_model_for_eap(bridge)

    with pytest.raises(NotImplementedError, match="hook surface required for EAP"):
        validate_model_for_eap(adapter)
