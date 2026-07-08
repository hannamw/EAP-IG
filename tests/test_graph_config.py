from types import SimpleNamespace

import pytest

from eap.graph import Graph


def test_graph_from_config_mapping_defaults_parallel_attn_mlp():
    graph = Graph.from_config({"n_layers": 2, "n_heads": 3, "d_model": 12})

    assert graph.cfg.n_layers == 2
    assert graph.cfg.n_heads == 3
    assert graph.cfg.d_model == 12
    assert graph.cfg.parallel_attn_mlp is False
    assert graph.n_forward == 1 + 2 * (3 + 1)
    assert graph.n_backward == 2 * (3 * 3 + 1) + 1
    assert "input" in graph.nodes
    assert "logits" in graph.nodes
    assert "a0.h0" in graph.nodes
    assert "m1" in graph.nodes


def test_graph_from_model_accepts_object_with_cfg():
    cfg = SimpleNamespace(
        n_layers=1,
        n_heads=2,
        d_model=8,
        parallel_attn_mlp=True,
        n_key_value_heads=None,
    )
    model = SimpleNamespace(cfg=cfg)

    graph = Graph.from_model(model)

    assert graph.cfg.n_layers == 1
    assert graph.cfg.n_heads == 2
    assert graph.cfg.parallel_attn_mlp is True


def test_graph_from_config_requires_core_shape_fields():
    with pytest.raises(ValueError, match="n_layers, n_heads, and d_model"):
        Graph.from_config({"n_layers": 1, "n_heads": 2})


@pytest.mark.parametrize(
    "extra",
    [
        {"is_stateful": True},
        {"is_multimodal": True},
        {"is_encoder_decoder": True},
        {"encoder_only": True},
        {"attention_dir": "bidirectional"},
        {"attn_only": True},
    ],
)
def test_graph_from_config_rejects_unsupported_bridge_shapes(extra):
    cfg = {"n_layers": 1, "n_heads": 2, "d_model": 8, **extra}

    with pytest.raises(NotImplementedError, match="decoder-only transformer"):
        Graph.from_config(cfg)


def test_graph_from_config_rejects_unresolved_gqa():
    cfg = {
        "n_layers": 1,
        "n_heads": 4,
        "n_key_value_heads": 2,
        "ungroup_grouped_query_attention": False,
        "d_model": 16,
    }

    with pytest.raises(NotImplementedError, match="grouped-query attention"):
        Graph.from_config(cfg)


def test_graph_from_config_accepts_ungrouped_gqa():
    cfg = {
        "n_layers": 1,
        "n_heads": 4,
        "n_key_value_heads": 2,
        "ungroup_grouped_query_attention": True,
        "d_model": 16,
    }

    graph = Graph.from_config(cfg)

    assert graph.cfg.n_heads == 4
