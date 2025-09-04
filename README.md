## Hierarchical Reasoning Model (HRM)

This is an implementation of the <a href="https://arxiv.org/abs/2506.21734">Hierachical Reasoning Model</a> (HRM) proposed by Guan Wang et al. I built it for educational purposes, with a few minor simplifications and extensions (see [Modifications to the Original Work](#modifications-to-the-original-work)).

The architecture is inspired by hierarchical, multi-timescale processing in the human brain. It uses two connected recurrent modules running at different frequencies:
- **H module** (slower): handles abstract planning
- **L module** (faster): handles low-level computations

Both are based on self-attention. Together, they enable reasoning in latent space.

## Hierarchical Reasoning Model (HRM)

This is an implementation of the <a href="https://arxiv.org/abs/2506.21734">Hierachical Reasoning Model</a> (HRM) proposed by Guan Wang et al. I built it for educational purposes, with a few minor simplifications and extensions (see [Modifications to the Original Work](#modifications-to-the-original-work)).

The architecture is inspired by hierarchical, multi-timescale processing in the human brain. It uses two connected recurrent modules running at different frequencies:
- **H module** (slower): handles abstract planning
- **L module** (faster): handles low-level computations

Both are based on self-attention. Together, this is an attempt to model reasoning in latent space.

---

👉 See [**Ablation Study: What Drives Performance?**](#ablation-study-what-drives-performance) for results on whether the H/L architecture is the main driver of HRM performance.

> **Note:** The ablation study section provides controlled experiments on the pathfinding task, examining segments, cycles, and architecture to understand what actually drives performance.

## Demo application

The model is applied to a pathfinding task: given an N×N board with obstacles, find the shortest path from START to END. The animation below shows actual inference steps as the model incrementally discovers the path.

<img width="300" src="https://github.com/user-attachments/assets/5bea57e8-5bec-4843-a945-25c49c0c4f1c" />

Legend: . = Floor, \# = Wall, S = Start point, E = End point, * = Path

## Usage

Dependencies: `Python`, `PyTorch` and `Pillow` <br/>
```bash
python -m pip install torch pillow
```

Train a model: <br/>
```bash
python boardpath.py --mode train
```

Run inference on a random board (also saves an animated GIF of the steps): <br/>
```
python boardpath.py --mode inference
```

To adjust the task, model, or training setup, edit `get_config()` and `get_train_config()` in `boardpath.py`. For example:
- Board size & obstacle density: `board_size`, `wall_prob`
- Embedding dimensionality (representation of each board cell): `d_model`
- \# of iterations ("think time"): `H_cycle_cnt`, `L_cycle_cnt`

All parameters are documented in `hrm/hrm.py`.

## Modifications to the Original Work

- Fixed "think time" (that is, fixed # of segments, both in train & inference) instead of Q-learning / ACT halting
- PyTorch SDPA instead of FlashAttention (same results, lower performance)
- RoPE and learned positional encodings can be used together
- Standard initialization instead of truncated LeCun normal
- Learnable (or fixed) initial H and L states
- `nn.RMSNorm` with learnable scale (`elementwise_affine=True`)
- Slight differences in where `nn.Dropout` is applied

## Model architecture

- `HRM` = `InputEmbedding` + two `ReasoningModule` instances (H and L) + linear projection (on last H state)
- `InputEmbedding` = `nn.Embedding` for tokens + optional absolute positional `nn.Embedding` (added to token embeddings)
- `ReasoningModule` = stack of `HRMBlock` instances
- `HRMBlock` has two sublayers:
  1. Attention: `SDPAttention` → `Dropout` → residual → `RMSNorm` (post-norm)
  2. MLP: `SwiGLU` → `Dropout` → residual → `RMSNorm`
- `SDPAttention` = scaled dot-product attention + linear projection, with optional RoPE

## Ablation Study: What Drives Performance?

Recent discussion around the Hierarchical Reasoning Model (HRM) raised a key question: **is the new two-timescale H/L architecture really the source of its strong performance?**

**TL;DR:** The main driver of performance is **training with more refinement segments**, not the H/L two-timescale split. Training with more segments teaches the model to refine its predictions when given extra inference steps. (See how this aligns study by the ARC Prize team, [link](#relation-to-arc-prize-analysis)).*

This analysis was carried out on a **board pathfinding task** (20×20 boards, wall probability = 0.3). Each variant was trained on **2000 training boards** and validated on **500 boards**, for **40 epochs** with **lr = 3e-4** and **batch size = 64**. Results are averaged across multiple runs.

- **Architecture details:**
  - Base HRM: *d_model = 256*, *4 heads*, *H/L blocks = 4*, *RoPE positional encoding*, *dropout = 0.1*.
  - H/L models: ~6.29M parameters.
  - Single-module models: ~6.23M parameters, matched by using *d_model = 360* instead of 256.

The results highlight key factors in this domain but should not be taken as comprehensive or conclusive evidence about the H/L architecture in general.

### What Was Varied
- **Architecture (variant):**
  - *H-only (detached):* single ReasoningModule unrolled for H×L steps, hidden state detached.
  - *H-only (bptt):* same as above, but with full backpropagation-through-time (BPTT).
  - *H/L:* full HRM with separate H and L modules.
- **Training segments (train_seg):** number of refinement segments during training.
- **Inference segments (infer_seg):** number of refinement segments during evaluation.
- **Cycles:** number of inner iterations (H×L).

### Results (all runs)

In the tables below:
- **board acc** = accuracy of predicting the entire board correctly.
- **acc4x** = the same metric, but with 4× more inference segments (extra test-time refinement).
- **Gap** = acc4x – board acc, showing how much accuracy improves with additional inference steps.

| Variant            | train_seg | infer_seg | H×L cycles | Best board acc | acc4x | Gap   |
|--------------------|-----------|-----------|------------|----------------|-------|-------|
| H-only (detached)  | 2         | 2         | 4          | 0.236          | 0.250 | +0.014|
| H-only (detached)  | 2         | 2         | 8          | 0.272          | 0.280 | +0.008|
| H-only (detached)  | 4         | 2         | 4          | 0.458          | 0.544 | +0.086|
| H-only (bptt)      | 2         | 2         | 4          | 0.274          | 0.336 | +0.062|
| H-only (bptt)      | 2         | 2         | 8          | 0.424          | 0.454 | +0.030|
| H-only (bptt)      | 4         | 2         | 4          | **0.616**      | **0.676** | +0.060|
| H/L                | 2         | 2         | 4          | 0.438          | 0.416 | -0.022|
| H/L                | 2         | 2         | 16         | 0.544          | 0.582 | +0.038|
| H/L                | 4         | 2         | 4          | 0.452          | 0.566 | +0.114|
| H/L                | 4         | 2         | 16         | 0.566          | 0.588 | +0.022|

### Aggregates
Results aggregated by dimension. (Averaging top-3 or last-5 epochs leads to the same conclusions.)

**By training segments**

| train_seg | Board acc (best) | Gap (best) |
|-----------|------------------|------------|
| 2         | 0.36             | +0.02      |
| 4         | **0.50**         | **+0.08**  |

**By architecture**

| Variant            | Board acc (best) | Gap (best) |
|--------------------|------------------|------------|
| H/L                | 0.48             | +0.04      |
| H-only (bptt)      | 0.44             | +0.05      |
| H-only (detached)  | 0.30             | +0.05      |

**By total cycles**

| Cycles | Board acc (best) | Gap (best) |
|--------|------------------|------------|
| 4      | 0.39             | **+0.06**  |
| 8      | 0.35             | +0.02      |
| 16     | **0.56**         | +0.03      |

### Iterative Refinement in Action
Training with more segments not only improves accuracy but also teaches the model to refine its predictions when given additional inference steps.

This effect is visible both in metrics and behavior:

- **Training curves:** With 2 training segments, board accuracy improves but acc and acc4x remain nearly identical. With 4 training segments, accuracy is higher *and* a clear gap opens up between acc and acc4x, showing the model learns to refine further at inference.

| Train Segments = 2 | Train Segments = 4 |
|--------------------|--------------------|
| ![](log_HL_hl22_t2_i2.png) | ![](log_HL_hl22_t4_i2.png) |

- **Animated demo:** An inference run shows the model solving a board step by step, refining its predictions across segments ([see GIF](boardpath.gif)).

The idea of iterative refinement is not entirely new — it has appeared in recurrent networks, diffusion models, and iterative decoding schemes. What is notable here is that the same effect emerges naturally when training HRM (or even single-module baselines) with multiple refinement segments.

### Conclusions
1. **Segments drive performance.** Using 4 segments instead of 2 leads to higher board accuracy and larger refinement gaps.
2. **Models learn to refine.** Training with more segments improves the ability to refine predictions when given extra inference steps.
3. **H/L is not the main driver.** The two-timescale architecture is competitive but does not outperform a single-module trained with BPTT.
4. **Cycles help to a point.** More inner iterations increase board accuracy somewhat, but refinement ability is strongest at moderate cycle counts.
5. **Robust across metrics.** These findings hold when using best, top-3, or last-5 epoch averages.

### Relation to ARC Prize Analysis
These results are consistent with the independent study by the ARC Prize team ([blog](https://arcprize.org/blog/hrm-analysis), [slides](https://docs.google.com/presentation/d/12IAuVKZXvbW6uCwzDhzN1PBh4fdjypjqyucYzoKJKMg/edit?slide=id.g32b23b2ea24_0_13#slide=id.g32b23b2ea24_0_13)). Their analysis on ARC tasks also found that outer-loop refinement drives performance, while the H/L split is not decisive. The pathfinding experiments here provide an additional supporting data point on a different benchmark, arriving at the same overall conclusions.
