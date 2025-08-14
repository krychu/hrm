import math
import torch
import torch.nn as nn
import pytest

from hrm.hrm import ReasoningModule  # adjust import

def make_module(d_model=64, nhead=4, dim_ff=128, num_layers=2, dropout=0.0):
    """Helper to build a small ReasoningModule with fixed seed for reproducible params."""
    torch.manual_seed(0)
    return ReasoningModule(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        num_layers=num_layers,
        dropout=dropout,
    )


def test_forward_shape_and_dtype():
    """
    Test: The module returns output with the same shape and dtype as the input.
    Checks basic interface correctness.
    """
    B, S, D = 3, 16, 64
    mod = make_module(d_model=D)
    x = torch.randn(B, S, D)
    inj = torch.randn(B, S, D)

    y = mod(x, inj)
    assert y.shape == (B, S, D)
    assert y.dtype == x.dtype


def test_injection_changes_output():
    """
    Test: Giving a non-zero input_injection changes the output compared to zero injection.
    This ensures the injection is actually being used in the forward pass.
    """
    B, S, D = 2, 8, 32
    mod = make_module(d_model=D)
    x = torch.randn(B, S, D)
    zero = torch.zeros_like(x)
    y_noinj = mod(x, zero)
    y_inj   = mod(x, torch.randn(B, S, D))
    # With high probability, they should differ
    assert not torch.allclose(y_noinj, y_inj)


def test_injection_associativity_equivalence():
    """
    Test: Because the forward adds `inputs + input_injection` once,
    mod(x, inj) should equal mod(x + inj, zeros).
    This checks that injection happens exactly once at the start.
    """
    B, S, D = 2, 12, 48
    mod = make_module(d_model=D)
    x   = torch.randn(B, S, D)
    inj = torch.randn(B, S, D)

    y1 = mod(x, inj)
    y2 = mod(x + inj, torch.zeros_like(inj))
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)


def test_gradients_flow_to_inputs_and_params():
    """
    Test: Gradients from a simple loss flow back to both the inputs and the module's parameters.
    Ensures the module is differentiable end-to-end.
    """
    B, S, D = 2, 10, 32
    mod = make_module(d_model=D)
    x = torch.randn(B, S, D, requires_grad=True)
    inj = torch.randn(B, S, D)

    y = mod(x, inj)                        # [B,S,D]
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    # At least one parameter should have grad
    assert any(p.grad is not None for p in mod.parameters())


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_dropout_determinism_in_eval(dropout):
    """
    Test: Dropout behaves as expected.
    - In train mode with dropout>0: two calls likely produce different outputs.
    - In eval mode: outputs are identical regardless of dropout setting.
    """
    B, S, D = 2, 8, 32
    mod = make_module(d_model=D, dropout=dropout)
    x = torch.randn(B, S, D)
    inj = torch.randn(B, S, D)

    # Train mode
    mod.train()
    y1 = mod(x, inj)
    y2 = mod(x, inj)
    if dropout > 0:
        assert not torch.allclose(y1, y2)
    else:
        assert torch.allclose(y1, y2)

    # Eval mode
    mod.eval()
    y3 = mod(x, inj)
    y4 = mod(x, inj)
    assert torch.allclose(y3, y4, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("S", [1, 4, 7, 16])
def test_various_sequence_lengths(S):
    """
    Test: Module handles different sequence lengths without shape errors.
    Useful if we later support puzzles of various sizes.
    """
    B, D = 2, 48
    mod = make_module(d_model=D)
    x   = torch.randn(B, S, D)
    inj = torch.randn(B, S, D)
    y = mod(x, inj)
    assert y.shape == (B, S, D)
