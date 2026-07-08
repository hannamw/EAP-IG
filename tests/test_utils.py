import torch
try:
    from transformer_lens.utilities import get_attention_mask
except ImportError:
    from transformer_lens.utils import get_attention_mask

from eap.utils import _attention_mask_from_tokens, _maybe_expand_grouped_query_tensor, tokenize_plus


class DummyCfg:
    def __init__(
        self,
        n_ctx=16,
        *,
        n_heads=None,
        n_key_value_heads=None,
        ungroup_grouped_query_attention=False,
    ):
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.n_key_value_heads = n_key_value_heads
        self.ungroup_grouped_query_attention = ungroup_grouped_query_attention


class DummyTokenizer:
    def __init__(
        self,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        padding_side="right",
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = padding_side


class DummyModel:
    def __init__(self, tokenizer):
        self.cfg = DummyCfg()
        self.tokenizer = tokenizer
        self._tokens = {
            "empty": [],
            "short": [11],
            "internal_pad": [12, tokenizer.pad_token_id, 13],
            "long": [21, 22, 23],
        }

    def to_tokens(self, inputs, prepend_bos=True, padding_side="right", truncate=False):
        assert prepend_bos is True
        assert padding_side == "right"
        sequences = []
        for item in inputs:
            seq = list(self._tokens[item])
            if prepend_bos:
                seq = [self.tokenizer.bos_token_id] + seq
            if truncate:
                seq = seq[: self.cfg.n_ctx]
            sequences.append(seq)
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            pad_count = max_len - len(seq)
            if padding_side == "right":
                padded.append(seq + [self.tokenizer.pad_token_id] * pad_count)
            else:
                padded.append([self.tokenizer.pad_token_id] * pad_count + seq)
        return torch.tensor(padded, dtype=torch.long)


def _assert_matches_transformer_lens(tokenizer, tokens, prepend_bos=True):
    torch.testing.assert_close(
        _attention_mask_from_tokens(tokenizer, tokens, prepend_bos=prepend_bos),
        get_attention_mask(tokenizer, tokens, prepend_bos),
    )


def test_local_attention_mask_matches_transformer_lens_right_padding():
    tokenizer = DummyTokenizer(
        pad_token_id=99,
        bos_token_id=99,
        eos_token_id=99,
        padding_side="right",
    )
    tokens = torch.tensor(
        [
            [99, 11, 99, 99],
            [99, 12, 99, 13],
            [99, 21, 22, 23],
        ],
        dtype=torch.long,
    )

    _assert_matches_transformer_lens(tokenizer, tokens)
    torch.testing.assert_close(
        _attention_mask_from_tokens(tokenizer, tokens),
        torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    )


def test_local_attention_mask_preserves_transformer_lens_bos_only_right_padding_behavior():
    tokenizer = DummyTokenizer(
        pad_token_id=99,
        bos_token_id=99,
        eos_token_id=99,
        padding_side="right",
    )
    tokens = torch.tensor([[99]], dtype=torch.long)

    _assert_matches_transformer_lens(tokenizer, tokens)
    torch.testing.assert_close(
        _attention_mask_from_tokens(tokenizer, tokens),
        torch.tensor([[0]], dtype=torch.long),
    )


def test_local_attention_mask_matches_transformer_lens_left_padding_bos_collision():
    tokenizer = DummyTokenizer(
        pad_token_id=99,
        bos_token_id=99,
        eos_token_id=99,
        padding_side="left",
    )
    tokens = torch.tensor(
        [
            [99, 99, 11],
            [99, 99, 99],
            [99, 12, 99],
        ],
        dtype=torch.long,
    )

    _assert_matches_transformer_lens(tokenizer, tokens)
    torch.testing.assert_close(
        _attention_mask_from_tokens(tokenizer, tokens),
        torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
            ],
            dtype=torch.long,
        ),
    )


def test_local_attention_mask_returns_all_ones_without_pad_token():
    tokenizer = DummyTokenizer(
        pad_token_id=None,
        bos_token_id=99,
        eos_token_id=99,
        padding_side="right",
    )
    tokens = torch.tensor([[99, 11, 99]], dtype=torch.long)

    torch.testing.assert_close(
        _attention_mask_from_tokens(tokenizer, tokens),
        torch.ones_like(tokens),
    )


def test_tokenize_plus_preserves_legacy_transformer_lens_mask_semantics():
    model = DummyModel(
        DummyTokenizer(
            pad_token_id=99,
            bos_token_id=99,
            eos_token_id=99,
            padding_side="right",
        )
    )

    tokens, attention_mask, input_lengths, n_pos = tokenize_plus(
        model,
        ["empty", "short", "internal_pad", "long"],
    )

    torch.testing.assert_close(
        attention_mask,
        get_attention_mask(model.tokenizer, tokens, True),
    )
    torch.testing.assert_close(
        attention_mask,
        torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(input_lengths, torch.tensor([0, 2, 4, 4]))
    assert n_pos == 4


def test_tokenize_plus_respects_max_length_and_restores_context():
    model = DummyModel(
        DummyTokenizer(
            pad_token_id=0,
            bos_token_id=99,
            eos_token_id=99,
            padding_side="right",
        )
    )
    model.cfg.n_ctx = 3

    tokens, attention_mask, input_lengths, n_pos = tokenize_plus(
        model,
        ["short", "long"],
        max_length=3,
    )

    assert model.cfg.n_ctx == 3
    torch.testing.assert_close(
        tokens,
        torch.tensor(
            [
                [99, 11, 0],
                [99, 21, 22],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(attention_mask, get_attention_mask(model.tokenizer, tokens, True))
    torch.testing.assert_close(input_lengths, torch.tensor([2, 3]))
    assert n_pos == 3


def test_expand_grouped_query_tensor_repeats_kv_heads_when_bridge_is_ungrouped():
    model = DummyModel(
        DummyTokenizer(
            pad_token_id=0,
            bos_token_id=99,
            eos_token_id=99,
            padding_side="right",
        )
    )
    model.cfg = DummyCfg(
        n_heads=4,
        n_key_value_heads=2,
        ungroup_grouped_query_attention=True,
    )
    tensor = torch.arange(1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)

    expanded = _maybe_expand_grouped_query_tensor(model, tensor)

    assert expanded.shape == (1, 2, 4, 3)
    torch.testing.assert_close(expanded[:, :, 0], tensor[:, :, 0])
    torch.testing.assert_close(expanded[:, :, 1], tensor[:, :, 0])
    torch.testing.assert_close(expanded[:, :, 2], tensor[:, :, 1])
    torch.testing.assert_close(expanded[:, :, 3], tensor[:, :, 1])
