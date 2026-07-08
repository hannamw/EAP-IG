import torch


def assert_strict_close(actual, expected, *, atol: float, name: str) -> None:
    diff = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
    max_abs_diff = diff.max().item() if diff.numel() else 0.0
    try:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"{name} strict parity failed: max_abs_diff={max_abs_diff:.6g}, atol={atol:.6g}"
        ) from exc
