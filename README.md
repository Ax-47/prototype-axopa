# Active JEPA

**An experimental Active Perception system that combines JEPA-style latent prediction with a Deep Q-Network (DQN) to recognize MNIST digits while revealing as few image regions as possible.**

Instead of seeing the entire image at once, the agent starts with a completely hidden image and decides — step by step — which region to reveal next. The model can also **imagine** what the complete image might look like before actually revealing a region.

```
             ┌──────────────────────┐
             │   Partial Observation│
             └──────────┬───────────┘
                         │
                         ▼
                    JEPA Encoder
                         │
                         ▼
                        z_t
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
- [Why JEPA + RL?](#why-jepa--rl)
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

**Active JEPA** instead frames recognition as a sequential decision problem. The agent learns two related things:

1. **What is the digit**, given the information I currently have?
2. **Which action should I take next** to reduce uncertainty as cheaply as possible?

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

### 1. JEPA Encoder

The encoder receives the currently visible image **and** the visibility mask, and produces a latent representation:

```
Partial Image ─────┐
                    ├──► Encoder ───► z_t
Mask ───────────────┘
```

- Latent dimension: **128**
- Implemented as convolutional layers followed by a linear projection.

### 2. Target Encoder

A separate target encoder is maintained as an **EMA (Exponential Moving Average)** of the online encoder, and receives no direct gradient updates:

```
Online Encoder ───────► z
       │
       │ EMA
       ▼
Target Encoder ───────► z_target
```

The JEPA objective pulls the predicted latent toward the target latent:

```
z_pred ≈ z_target
```

### 3. Action Encoder

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

This lets the predictor model *"what will my latent representation look like if I reveal this particular cell?"*

### 4. JEPA Predictor

The predictor takes the current latent and an action, and predicts the resulting latent:

```
(z_t, action) ───► Predictor ───► z_pred
```

Conceptually:

> "I currently see this." + "I am going to reveal cell 4." → "This is what I expect to know afterwards."

### 5. Classifier

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

### 6. Full Image Decoder

The predictor's latent can also be decoded into an *imagined* complete image:

```
z_pred
  │
  ▼
Full Image Decoder
  │
  ▼
28 × 28 imagined image
```

During inference, three images can be displayed side by side:

| Current Observation | JEPA Imagination | Real Full Image |
|---|---|---|
| visible regions only | predicted image | ground truth (debug only) |

---

## DQN

The JEPA model supplies information about the current state; the DQN decides what to do with it.

**State composition:**

| Component | Dim |
|---|---|
| Latent representation | 128 |
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

1. **Encode the current observation** → partial image + mask → `z_t`
2. **Predict the current digit** → `z_t` → classifier → probabilities
3. **Imagine every possible cell**:
   ```
   (z_t, cell 0) → z_pred → classify
   (z_t, cell 1) → z_pred → classify
   ...
   (z_t, cell 8) → z_pred → classify
   ```
   giving the DQN a look-ahead over possible future states.
4. **Construct the state**: `z` + current prediction + confidence + entropy + opened mask + candidate predictions
5. **DQN selects an action** → reveal cell, or STOP
6. **If a cell is selected**, the environment reveals it and the loop repeats.
7. **If STOP is selected**, the episode ends and the digit prediction is evaluated.

---

## Training Objectives

The JEPA model is trained with multiple complementary losses.

**JEPA loss** — predictor matches the target latent:
```
L_jepa = SmoothL1(z_pred, z_target)
```

**Classification loss** — applied to both current and predicted latents:
```
L_cls = CrossEntropy(logits, label)
```

**Partial reconstruction** — decode the next partial observation:
```
L_partial = MSE(reconstruction, next_image)
```

**Full image reconstruction** — decode the complete original image:
```
L_full = MSE(full_reconstruction, full_image)
```

**Total loss:**
```
L = 1.0 × L_jepa
  + 1.0 × L_classification
  + 0.1 × L_partial
  + 0.5 × L_full
```

The reconstruction losses mainly provide extra learning signal and make the latent space more interpretable.

---

## Reinforcement Learning

The DQN is trained to minimize unnecessary reveals while still identifying the digit correctly.

| Event | Reward |
|---|---|
| Reveal cell | −0.10 |
| Correct STOP | +1.00 |
| Wrong STOP | −1.00 |

The goal isn't just *"recognize MNIST"* — it's:

> **"Recognize MNIST using as few observations as possible."**

---

## Inference

```bash
uv run play-jepa
```

The program continuously samples random MNIST test images and, for each one, repeats:

```
Start with nothing → Choose cell → Reveal → Update belief → Choose cell / STOP → repeat
```

After STOP, it immediately moves on to the next image. Stop the program with `Ctrl+C`.

### Example Output

```
============================================================
IMAGE #1
MNIST index: 3842
Real label: 7
============================================================

Step 1
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

>>> STOP

Prediction: 7
Real label: 7
Correct: True
Opened 2/9 cells
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
│       ├── model.py
│       ├── train.py
│       ├── train_rl.py
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

### Train DQN

After JEPA training:

```bash
uv run python -m active_jepa.train_rl
```

Produces: `checkpoints/rl_v2.pt`

---

## Running the Agent

```bash
uv run play-jepa
```

The agent continuously plays random MNIST test images. Press `Ctrl+C` to stop.

---

## Why JEPA + RL?

JEPA and RL solve different parts of the problem:

**JEPA** learns *"what do I expect to see?"* and *"what does this partial observation represent?"*

**RL** learns *"what should I look at next?"*

```
        JEPA
         │
   understand / predict
         │
         ▼
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

This separates **representation learning** from **decision making**.

---

## Future Work

- Better information-gain based action selection
- Multi-step JEPA prediction
- Hierarchical JEPA
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

- Joint Embedding Predictive Architectures (JEPA)
- Representation learning
- Active perception
- Reinforcement learning (DQN)
- Latent-space prediction
- Visual imagination

The current environment is intentionally simple — MNIST + 3×3 region selection. The interesting part isn't MNIST classification itself, but **whether an agent can learn when and where to look**.
