import math
import torch
import torch.nn as nn
import pytest
from hrm.hrm import InputEmbedding

@pytest.mark.parametrize("batch,seq_len,vocab,dim", [(4,16,6,32), (2,16,6,128)])
def test_shapes_and_broadcast(batch, seq_len, vocab, dim):
    """
    Ensures output shapes are [B, S, D] and that adding [S, D] positional
    embeddings broadcasts correctly against [B, S, D] token embeddings.
    """
    torch.manual_seed(0)
    emb = InputEmbedding(vocab_cnt=vocab, seq_len=seq_len, embedding_dim=dim)
    x = torch.randint(low=0, high=vocab, size=(batch, seq_len), dtype=torch.long)
    out = emb(x)
    assert out.shape == (batch, seq_len, dim)

    # Compare to explicit unsqueeze version to prove equivalence
    x_tok = emb.tok(x)                               # [B, S, D]
    x_pos_unsq = emb.pos(emb.pos_idx).unsqueeze(0)   # [1, S, D]
    out2 = emb.scale * (x_tok + x_pos_unsq)
    torch.testing.assert_close(out, out2, rtol=1e-6, atol=1e-6)

def test_grad_flows():
    """
    Verifies that both token and positional embeddings receive gradients.
    """
    torch.manual_seed(0)
    emb = InputEmbedding(vocab_cnt=6, seq_len=16, embedding_dim=32)
    x = torch.randint(0, 6, (3, 16), dtype=torch.long)
    out = emb(x).sum()
    out.backward()

    assert emb.tok.weight.grad is not None
    assert emb.pos.weight.grad is not None
    assert emb.tok.weight.grad.shape == emb.tok.weight.shape
    assert emb.pos.weight.grad.shape == emb.pos.weight.shape


def test_determinism_cpu():
    """
    With the same RNG seed and constructor args, two fresh models should
    produce identical outputs (checks init + forward determinism on CPU).
    """
    torch.manual_seed(123)
    emb1 = InputEmbedding(vocab_cnt=6, seq_len=16, embedding_dim=32)
    x = torch.randint(0, 6, (2, 16), dtype=torch.long)
    y1 = emb1(x).detach().clone()

    torch.manual_seed(123)  # reset to ensure same init for the next model
    emb2 = InputEmbedding(vocab_cnt=6, seq_len=16, embedding_dim=32)
    y2 = emb2(x).detach().clone()

    torch.testing.assert_close(y1, y2, rtol=1e-6, atol=1e-6)

def test_position_effect_is_consistent():
    """
    If all tokens are identical, differences between positions should equal
    the positional embedding differences (times the √D scale).
    """
    torch.manual_seed(0)
    emb = InputEmbedding(vocab_cnt=6, seq_len=16, embedding_dim=32)

    # All tokens identical → token embedding cancels out in positional diffs
    x = torch.ones((1, 16), dtype=torch.long)
    y = emb(x)[0]  # [S, D]

    scale = emb.scale
    pos = emb.pos(emb.pos_idx)  # [S, D]
    diff_model = y[7] - y[3]
    diff_expected = scale * (pos[7] - pos[3])
    torch.testing.assert_close(diff_model, diff_expected, rtol=1e-6, atol=1e-6)
