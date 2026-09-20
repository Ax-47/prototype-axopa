# Active JEPA

**An experimental Active Perception system that combines a *hierarchical* JEPA-style latent predictor with a Deep Q-Network (DQN) to recognize MNIST digits while revealing as few image regions as possible.**

Instead of seeing the entire image at once, the agent starts with a completely hidden image and decides — step by step — which region to reveal next. The model predicts what it expects to happen at **two levels of abstraction** at once: a coarse, scene-level "what digit is this" belief, and a fine-grained, patch-level "what will this specific region look like" belief. It can **imagine** both before actually revealing a region.

```
             ┌──────────────────────┐
             │   Partial Observation│
             └──────────┬───────────┘
                         │
                         ▼
              Hierarchical JEPA Encoder
                 (local patches → global scene)
                         │
                         ▼
                  z_t  (global latent)
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        Classifier                 DQN
             │                       │
             ▼                       ▼
       Current Digit           Choose Action
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                       Reveal cell             STOP
                          │
                          ▼
                   New Observation
                          │
                          └───────► repeat
```

---

## Table of Contents

- [Overview](#overview)
- [Active Perception](#active-perception)
- [Architecture](#architecture)
- [DQN](#dqn)
- [Decision Process](#decision-process)
- [Training Objectives](#training-objectives)
- [Reinforcement Learning](#reinforcement-learning)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
- [Running the Agent](#running-the-agent)
- [Why Hierarchical JEPA + RL?](#why-hierarchical-jepa--rl)
- [Future Work](#future-work)
- [Status](#status)

---

## Overview

Traditional MNIST classification gives the model the entire image at once:

```
Full Image
    │
    ▼
 Encoder
    │
    ▼
 Classifier
    │
    ▼
  Digit
```

**Active JEPA** instead frames recognition as a sequential decision problem, and additionally frames *perception itself* as a two-level (hierarchical) problem. The agent learns three related things:

1. **What is the digit**, given the information I currently have? (global level)
2. **What will this specific patch look like** once I reveal it? (local level)
3. **Which action should I take next** to reduce uncertainty as cheaply as possible? (RL)

---

## Active Perception

The 28×28 MNIST image is divided into a 3×3 grid:

```
┌─────┬─────┬─────┐
│  0  │  1  │  2  │
├─────┼─────┼─────┤
│  3  │  4  │  5  │
├─────┼─────┼─────┤
│  6  │  7  │  8  │
└─────┴─────┴─────┘
```

The agent has **10 possible actions**:

| Action | Meaning |
|--------|---------|
| 0–8    | Reveal that cell |
| 9      | STOP |

At the beginning of every episode, nothing is visible:

```
┌─────┬─────┬─────┐
│  ?  │  ?  │  ?  │
├─────┼─────┼─────┤
│  ?  │  ?  │  ?  │
├─────┼─────┼─────┤
│  ?  │  ?  │  ?  │
└─────┴─────┴─────┘
```

The model chooses a cell, reveals it, updates its belief, and chooses again. The episode ends when the model chooses **STOP**.

---

## Architecture

The encoder is **hierarchical**: a local (patch-level) representation is built first, then aggregated into a global (scene-level) representation. Prediction, targets, and decoding all happen at *both* levels.

### 1. Local Encoder (Level 0)

A conv trunk turns the visible image + mask into a **7×7 grid of 49 patch tokens** (128-dim each), then normalizes them:

```
Partial Image ─────┐
                    ├──► Conv trunk ───► 49 patch tokens ───► LayerNorm ───► tokens
Mask ───────────────┘
```

The `LayerNorm` matters: raw conv activations are unbounded, and without normalizing them here, the local predictor/decoder downstream can saturate.

### 2. Global Aggregator (Level 1)

A small Transformer (with a learned CLS token) aggregates the 49 local tokens into a single 128-dim **global latent** `z`:

```
tokens ──► + positional embedding ──► Transformer (2 layers) ──► CLS output ──► z
```

The Transformer's internal LayerNorm keeps `z` naturally well-scaled.

### 3. Target Encoder

A separate target encoder (covering *both* levels — local encoder + global aggregator) is maintained as an **EMA (Exponential Moving Average)** of the online encoder, and receives no direct gradient updates:

```
Online Encoder (local + global) ───────► z, tokens
              │
              │ EMA
              ▼
Target Encoder (local + global) ───────► z_target, tokens_target
```

The JEPA objective pulls each predicted latent toward its corresponding target latent, at both levels:

```
z_pred      ≈ z_target
tokens_pred ≈ tokens_target
```

### 4. Action Encoder

Actions are represented via an embedding:

```
action
  │
  ▼
Embedding
  │
  ▼
action representation
```

This lets both predictors model *"what will my representation look like if I reveal this particular cell?"* — at the scene level and at the patch level.

### 5. JEPA Predictors (two, one per level)

**Global predictor** — same role as before, operating on the scene-level latent:

```
(z_t, action) ───► Global Predictor ───► z_pred
```

**Local predictor** — new: operates on the 49 patch tokens, broadcasting the action embedding to every token and applying a residual + LayerNorm:

```
(tokens_t, action) ───► Local Predictor ───► tokens_pred
```

Conceptually:

> "I currently see this." + "I am going to reveal cell 4." → "This is what I expect to know about the whole scene, **and** what I expect each patch to look like, afterwards."

### 6. Classifier

Operates on the global latent, unchanged in spirit:

```
z
│
▼
MLP
│
▼
10 logits
│
▼
digit probability
```

```
prediction = probabilities.argmax()
confidence = probabilities.max()
```

The classifier can estimate the current digit **without** requiring the full image.

### 7. Decoders (two, one per level)

**Local decoder** (`SpatialDecoder`) — reshapes the predicted patch tokens straight back into their native 7×7 spatial layout (no vector round-trip needed, since tokens already carry spatial structure) and decodes the **next partial observation**:

```
tokens_pred
  │
  ▼
reshape to (128, 7, 7)
  │
  ▼
Deconv stack
  │
  ▼
28 × 28 imagined partial observation
```

**Global decoder** (`FullImageDecoder`) — decodes the predicted global latent into the **complete hidden image**, unchanged from before:

```
z_pred
  │
  ▼
Full Image Decoder
  │
  ▼
28 × 28 imagined full image
```

During inference, **four** images are displayed side by side:

| Current Observation | Local Decoder (patch-level) | Global Decoder (scene-level) | Real Full Image |
|---|---|---|---|
| visible regions only | predicted next partial obs | predicted complete digit | ground truth (debug only) |

---

## DQN

The JEPA model supplies information about the current state; the DQN decides what to do with it. The state is built **only from the global latent** — the local patch tokens are used internally by the hierarchical predictor but are not fed into the DQN, so `state_dim` is unchanged from the non-hierarchical version.

**State composition:**

| Component | Dim |
|---|---|
| Global latent representation | 128 |
| Class probabilities | 10 |
| Current entropy | 1 |
| Current confidence | 1 |
| Opened ratio | 1 |
| Opened mask | 9 |
| Candidate features (entropy + confidence × 9 cells) | 18 |
| **Total** | **168** |

**Network:**

```
168 → 256 → 256 → 10
```

Output = one Q-value per action:

```
Q[0..8] = reveal cell
Q[9]    = STOP
```

Action selection: `action = q.argmax()`

---

## Decision Process

At every step:

1. **Encode the current observation** → partial image + mask → `z_t` (global) + `tokens_t` (local)
2. **Predict the current digit** → `z_t` → classifier → probabilities
3. **Imagine every possible cell**, at both levels:
   ```
   (z_t, tokens_t, cell 0) → z_pred, tokens_pred → classify
   (z_t, tokens_t, cell 1) → z_pred, tokens_pred → classify
   ...
   (z_t, tokens_t, cell 8) → z_pred, tokens_pred → classify
   ```
   giving the DQN a look-ahead over possible future states.
4. **Construct the state**: `z` + current prediction + confidence + entropy + opened mask + candidate predictions
5. **DQN selects an action** → reveal cell, or STOP
6. **If a cell is selected**, the environment reveals it and the loop repeats.
7. **If STOP is selected**, the episode ends and the digit prediction is evaluated.

---

## Training Objectives

The JEPA model is trained with multiple complementary losses, now split across both levels of the hierarchy.

**JEPA loss** — predictor matches the target latent, at both levels:
```
L_jepa_global = SmoothL1(z_pred, z_target)
L_jepa_local  = SmoothL1(tokens_pred, tokens_target)
L_jepa        = 0.5 × L_jepa_global + 0.5 × L_jepa_local
```

**Classification loss** — applied to both current and predicted global latents:
```
L_cls = CrossEntropy(logits, label)
```

**Partial reconstruction** — decode the next partial observation from the local tokens:
```
L_partial = BCE(reconstruction, next_image)
```

**Full image reconstruction** — decode the complete original image from the global latent:
```
L_full = BCE(full_reconstruction, full_image)
```

> **Why BCE instead of MSE?** Both decoders end in a `Sigmoid`. MSE's gradient with respect to the pre-`Sigmoid` value carries an extra `sigmoid'(z)` factor that vanishes once the output saturates near 0 or 1 — so a decoder that starts predicting "all black" (a fairly good local minimum on MNIST, since most of the image is background) can get stuck there, with almost no gradient pushing it back out. BCE's gradient reduces to `(prediction - target)` with no vanishing factor, so it doesn't have this failure mode.

**Total loss:**
```
L = 1.0 × L_jepa
  + 1.0 × L_classification
  + 0.5 × L_partial
  + 0.5 × L_full
```

The reconstruction losses mainly provide extra learning signal and make the latent space more interpretable — but they're also the *only* signal that trains the two decoder heads directly, which is why their weights are kept comparable to each other.

---

## Reinforcement Learning

The DQN is trained to minimize unnecessary reveals while still identifying the digit correctly.

| Event | Reward |
|---|---|
| Reveal cell | −0.20 |
| Correct STOP | +1.00 |
| Wrong STOP | −1.00 |

> **Why 0.20 instead of 0.10?** With the original cost, revealing one more cell only needed to improve accuracy by `reveal_cost / (correct_reward - wrong_reward) = 0.10 / 2.0 = 5%` to be "worth it" — a bar so low that the agent almost always kept opening cells instead of stopping early. Raising the cost to 0.20 raises that breakeven to ~10%, pushing the agent toward stopping sooner once it's genuinely confident.

**Exploration also anneals a second knob**, `stop_exploration_probability` — the chance of forcing a random STOP action during ε-greedy exploration. It starts high (`0.60`) early in training so the replay buffer collects enough early-STOP examples for the Q-network to actually learn that stopping early can be good, then decays to `0.20` as training progresses and the policy becomes more competent.

The goal isn't just *"recognize MNIST"* — it's:

> **"Recognize MNIST using as few observations as possible."**

---

## Inference

```bash
uv run play-jepa
```

The program opens one matplotlib window and plays through MNIST test images **automatically, one after another** — as soon as an image ends (via STOP or hitting the step cap), the next random image starts immediately in the same window. Close the window (or hit `Ctrl+C`) to stop.

Four panels are shown for every step:

```
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│ Current Observation │   Local Decoder     │   Global Decoder    │   Real Full Image   │
│  (visible so far)    │ (tokens → next obs) │  (z → full digit)   │   (ground truth)     │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘
```

### Example Output

```
[Image 1] Step 1
Opened: []
Current prediction: 3
Confidence: 0.4213
Entropy: 1.8921

Candidate cells:
  cell 0: Q=+0.1821 | digit=3 | conf=0.51 | entropy=1.42
  cell 1: Q=+0.2914 | digit=7 | conf=0.72 | entropy=0.91
  cell 2: Q=+0.1032 | digit=3 | conf=0.48 | entropy=1.51

  STOP: Q=-0.4521

>>> Reveal cell 1

...

[Image 1] STOP | prediction=7 | real=7 | opened=2/9

[Image 2] Step 1
...
```

---

## Project Structure

```
active-jepa/
│
├── checkpoints/
│   ├── latest.pt
│   └── rl_v2.pt
│
├── data/
│
├── src/
│   └── active_jepa/
│       ├── data.py
│       ├── model.py         # hierarchical JEPA (local + global)
│       ├── environment.py   # Gym-style RL environment, wraps the JEPA model
│       ├── train.py         # trains the JEPA (checkpoints/latest.pt)
│       ├── train_rl.py      # trains the DQN (checkpoints/rl_v2.pt)
│       ├── play.py
│       └── rl.py
│
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python >= 3.11
- PyTorch
- Torchvision
- tqdm
- CUDA supported when a compatible NVIDIA environment is available

The project uses **uv** for Python environment and dependency management.

---

## Installation

```bash
git clone <repo-url>
cd active-jepa
uv sync
```

---

## Training

### Train JEPA

```bash
uv run train-jepa
```

Produces the JEPA checkpoint: `checkpoints/latest.pt`

> Any time `model.py` changes (a new layer, a different loss target shape, etc.), this checkpoint is no longer compatible with the new architecture and must be regenerated from scratch.

### Train DQN

After JEPA training:

```bash
uv run train-rl
```

Produces: `checkpoints/rl_v2.pt`

> This only needs to be re-run when the JEPA encoder's weights change (a new `checkpoints/latest.pt`) or when the reward/exploration hyperparameters in `train_rl.py` change — not when unrelated files are edited. The DQN operates on a fixed 168-dim state vector, so its architecture itself is stable across JEPA changes; it's the *meaning* of that state vector that shifts whenever the JEPA encoder is retrained.

---

## Running the Agent

```bash
uv run play-jepa
```

The agent continuously plays through random MNIST test images, one after another, in a single window. Close the window or press `Ctrl+C` to stop.

---

## Why Hierarchical JEPA + RL?

JEPA and RL solve different parts of the problem, and the hierarchy splits the JEPA half further:

**Local JEPA level** learns *"what will this specific patch look like once revealed?"*

**Global JEPA level** learns *"what digit is this, given everything seen so far?"*

**RL** learns *"what should I look at next?"*

```
              Local JEPA               Global JEPA
                  │                         │
          patch-level prediction   scene-level prediction
                  │                         │
                  └────────────┬────────────┘
                                │
                              State
                                │
                                ▼
                               DQN
                                │
                         choose action
                                │
                                ▼
                        reveal / STOP
```

This separates **representation learning** (now itself split across two levels of abstraction) from **decision making**.

---

## Future Work

- Better information-gain based action selection
- Multi-step JEPA prediction
- ~~Hierarchical JEPA~~ ✅ done — local (patch) + global (scene) levels
- A third, even coarser level (e.g. digit-family clusters) for a deeper hierarchy
- Vectorized environments for faster RL training
- Cached JEPA representations
- Larger image observation spaces
- More complex environments than MNIST
- Vision Transformer encoder
- Transformer-based predictor
- Actor-Critic instead of DQN
- Curiosity / intrinsic reward
- Learned stopping criterion
- Comparing active perception against random cell selection
- Measuring accuracy versus number of revealed cells

---

## Status

This is an experimental research/learning project exploring the combination of:

- Hierarchical Joint Embedding Predictive Architectures (JEPA)
- Representation learning at multiple levels of abstraction
- Active perception
- Reinforcement learning (DQN)
- Latent-space prediction
- Visual imagination

The current environment is intentionally simple — MNIST + 3×3 region selection. The interesting part isn't MNIST classification itself, but **whether an agent can learn when and where to look, and whether local and global beliefs can be learned and predicted together.**
