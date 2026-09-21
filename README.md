# Active JEPA

**An experimental Active Perception system that combines a *hierarchical* JEPA-style latent predictor with a Deep Q-Network (DQN) to recognize MNIST digits while revealing as few image regions as possible.**

Instead of seeing the entire image at once, the agent starts with a completely hidden image and decides — step by step — which region to reveal next, and when to stop. The model keeps a coarse, scene-level "what digit is this" belief and a fine-grained, patch-level "what will this region look like" belief, and it learns how much each possible reveal is *worth*.

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
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   Classifier     AccuracyHead          DQN
        │               │                │
        ▼               ▼                ▼
  Current Digit   P(correct) after   Choose Action
                  each possible          │
                  reveal          ┌──────┴──────┐
                                Reveal cell    STOP
                                  │
                                  ▼
                           New Observation ──► repeat
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
- [Evaluation](#evaluation)
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

Traditional MNIST classification gives the model the entire image at once. **Active JEPA** instead frames recognition as a sequential decision problem, and frames *perception itself* as a multi-level problem. The system learns four related things:

1. **What is the digit**, given what I have seen so far? (global level, classifier)
2. **What will this patch look like** once revealed? (local / mid levels, JEPA predictors + decoders)
3. **How likely am I to be right** if I stop now, or after revealing cell *c*? (AccuracyHead)
4. **Which action should I take next**? (RL)

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

At the beginning of every episode nothing is visible. The episode ends when the agent chooses **STOP**, or after all 9 cells have been revealed (the current belief is then graded).

---

## Architecture

The encoder is **hierarchical**: local patch tokens → region tokens → one global latent. Prediction, EMA targets and decoding happen at *all three* levels.

### 1. Local Encoder (Level 0)

A conv trunk turns the visible image + mask into a **7×7 grid of 49 patch tokens** (128-dim each), then `LayerNorm`s them (raw conv activations are unbounded and can saturate the decoders' `Sigmoid`).

### 2. Mid Pooler (Level 1)

The 49 local tokens are adaptively average-pooled to a **3×3 grid of 9 region tokens** — the same grid the reveal actions index into.

### 3. Global Aggregator (Level 2)

A small Transformer (learned CLS token) aggregates the 9 region tokens into one 128-dim **global latent** `z`. It uses **no dropout**: dropout used to be on by default, and because `model.train()` also switched the target encoder to train mode, the EMA targets were noisy.

### 4. Target Encoder

An EMA copy of the whole encoder (cosine momentum schedule, BYOL-style). It is **always kept in eval mode** (`ActiveJEPA.train()` re-evals it) and receives no gradients. The JEPA objective pulls each predicted latent toward its target at every level:

```
z_pred ≈ z_target      tokens_mid_pred ≈ tokens_mid_target      tokens_local_pred ≈ tokens_local_target
```

### 5. Action Encoder

An embedding of the 10 actions, fed to all predictors.

### 6. JEPA Predictors (three, one per level)

Global, mid and local predictors each predict a residual, then `LayerNorm`. Mid/local predictors get a **position gate** saying which tokens the reveal can change, plus **top-down FiLM** conditioning (global → mid → local). STOP is an exact identity map at every level.

The gates are computed from the cell's **true pixel footprint**, dilated by `gate_dilation` (default 1) local tokens, because the conv trunk's receptive field is ~18 px and the mid pooler's windows overlap — revealing one cell changes tokens around it too. A hard one-hot gate could not represent that.

### 7. Classifier

```
z → MLP → 10 logits → probabilities
```

### 8. AccuracyHead

```
(z.detach(), opened-cell mask) → MLP → 10 logits → sigmoid
   [0..8] = P(classifier is correct) AFTER revealing that cell
   [9]    = P(classifier is correct) if we STOP right now
```

This is the estimate of the **value of information** that the RL state and the greedy planner use. It is trained with BCE against whether the classifier is *actually correct* on the real next observation.

> **Why not classify the imagined `z_pred`?** That was the previous design (imagined entropy / confidence per cell). But `z_pred` is a regression-to-the-mean latent that is a function of `z` and the action only — it carries no information about the *unseen* cell's content — and `softmax(classifier(E[z']))` is not `E[softmax(classifier(z'))]` (Jensen gap). The imagined confidences therefore tracked the current confidence rather than how much a reveal would help. The head reads `z.detach()`, so it never bends the representation; it only learns to read it.

### 9. Decoders (three, one per level)

- **Local decoder** (`SpatialDecoder`): predicted local tokens → next partial observation (finest).
- **Mid decoder** (`MidDecoder`): predicted region tokens → next partial observation (coarser).
- **Global decoder** (`FullImageDecoder`): predicted global latent → the complete hidden image.

`play.py` shows five panels: current observation, local / mid / global decoder output for the chosen action, and the real image (debug only).

---

## DQN

The state is built only from the global latent and the classifier/head outputs. The layout is defined **once**, in `environment.py` (`PROB_SLICE`, `OPENED_MASK_SLICE`, …, `STATE_DIM`) — nothing else hard-codes offsets.

| Component | Dim |
|---|---|
| Global latent `z` | 128 |
| Class probabilities | 10 |
| Current entropy | 1 |
| Current confidence | 1 |
| Opened ratio | 1 |
| Opened mask | 9 |
| Candidate features (per cell: P(correct) after reveal, and its gain over P(correct) now; 0 if open) | 18 |
| **Total** | **168** |

**Network:** `168 → 256 → 256 → 10`, one Q-value per action; invalid (already-open) reveals are masked to `-inf`. Trained with **Double DQN**.

---

## Decision Process

At every step:

1. **Encode** the partial image + mask → `z_t`, `tokens_mid`, `tokens_local`.
2. **Classify** → probabilities, entropy, confidence.
3. **Estimate the value of every action** with the AccuracyHead: P(correct) after each unopened cell, and P(correct) if we stop now.
4. **Build the state** and let the DQN choose (or, in `play.py`, let the greedy planner choose and use the DQN only to break near-ties — see `planner.py`).
5. **Reveal** the cell and repeat, or **STOP** and grade the prediction.

---

## Training Objectives

**Data (`ActiveMNIST`).** A random 0–9 cells are already open (so a fully revealed image is seen as *input* too — the RL episode ends in exactly that state) and one more is revealed; if all 9 are open the action is STOP and the observation is unchanged. Training uses a fixed **50k / 8k / 2k split** of MNIST-train (`data.split_indices`): the JEPA trains on the first part, the DQN on the second, validation uses the third. The real test set is only used by `eval_rl.py` and `play.py`.

**JEPA loss** — all three levels, smooth-L1 against the EMA target:
```
L_jepa = (1/3) L_global + (1/3) L_mid + (1/3) L_local
```

**Classification** — on the current and the predicted global latent:
```
L_cls = 0.5 CE(logits_current, y) + 0.5 CE(logits_pred, y)
```

**Accuracy head** — BCE on the real outcome (the head input is detached):
```
L_voi = BCE(head[action], 1[classifier correct on next obs]) + BCE(head[STOP], 1[classifier correct now])
```

**Reconstruction** — BCE (not MSE, which vanishes once a Sigmoid saturates):
```
L_partial = BCE(local_decoder(tokens_local_pred), next_image)
L_mid     = BCE(mid_decoder(tokens_mid_pred),     next_image)
L_full    = BCE(full_decoder(z_pred),             full_image)
```

**Total:**
```
L = 1.0 L_jepa + 1.0 L_cls + 0.5 L_voi + 0.5 L_partial + 0.3 L_mid + 0.5 L_full
```

After every epoch `train.py` reports held-out classifier accuracy (current / predicted latent) and the head's BCE.

---

## Reinforcement Learning

The DQN learns to recognize MNIST with as few reveals as possible.

| Event | Reward |
|---|---|
| Reveal cell | −0.20 |
| Correct STOP (or graded at 9 cells) | +1.00 |
| Wrong STOP (or graded at 9 cells) | −1.00 |

**Breakeven:** one more reveal is worth it only if it raises expected accuracy by more than `reveal_cost / (correct_reward − wrong_reward) = 0.20 / 2.0 = 10%`. That analysis is now *actually true*: the old entropy-drop bonus (`0.5 · max(ΔH, 0)`, reveals only) subsidised every reveal relative to STOP — a ~0.4-nat entropy drop paid the entire 0.20 reveal cost, whether or not the digit was recognised. It is **off by default** (`info_gain_weight=0.0`). If you turn it on it is now *potential-based* (`Φ(s) = −w·H(s)`, terminal Φ = 0, γ = 1), which telescopes to a constant per episode and therefore cannot change the optimal policy.

**Exploration** is ε-greedy with a uniform choice over *all* valid actions **including STOP**, so exploratory episodes end at each depth 0–9 with equal probability. (The previous "force STOP with probability 0.6·ε" made ~60% of early episodes stop with nothing revealed and almost none reach 5+ cells.)

**Other details:** the replay buffer is a preallocated ring buffer on the training device; one gradient step per 4 environment steps after 5 000 warm-up steps; the JEPA state is computed once per step (it used to be computed twice); the DQN trains on the held-out `rl_train` split and is validated on `rl_val`.

---

## Evaluation

```bash
uv run eval_rl --episodes 1000
```

On the MNIST **test** set, it reports accuracy, average cells revealed and average return for:

| Policy | Meaning |
|---|---|
| `dqn` | trained Q-network, greedy |
| `hybrid (greedy+dqn)` | expected-value planner, DQN breaks ties (what `play.py` runs) |
| `greedy (head only)` | the planner alone |
| `fixed@t` / `random@t` | reveal in a fixed (centre-first) / random order until confidence ≥ t |

If the DQN cannot beat the best `fixed@t` / `random@t` row, it has learned *when to stop* but not *where to look*. It also prints a calibration report for the AccuracyHead and the raw softmax confidence (mean prediction vs. actual accuracy, and a Brier skill score against the base rate).

`train_rl.py` additionally evaluates greedily on `rl_val` every 5 000 episodes and keeps the best checkpoint as `checkpoints/rl_v2_best.pt`.

---

## Inference

```bash
uv run play-jepa
```

Plays MNIST test images automatically in one matplotlib window until you close it. At every step it prints the value and P(correct) of every candidate action:

```
[Image 1] Step 2
Opened: [4]
Current prediction: 7
Confidence: 0.6120
Entropy: 1.1034

Candidate values:
  cell 1: value=+0.3120 P(correct)=0.756 (tied) <-- chosen
  cell 3: value=+0.2980 P(correct)=0.749 (tied)
  STOP: value=+0.0100 P(correct)=0.505
  ...
```

---

## Project Structure

```
active-jepa/
│
├── checkpoints/
│   ├── latest.pt          # JEPA (+ classifier, AccuracyHead)
│   ├── rl_v2.pt           # DQN, latest
│   └── rl_v2_best.pt      # DQN, best on rl_val
│
├── data/
│
├── src/
│   └── active_jepa/
│       ├── data.py          # dataset, 50k/8k/2k split, reveal_cells
│       ├── model.py         # hierarchical JEPA + AccuracyHead
│       ├── environment.py   # RL environment, shared build_state, state layout
│       ├── rl.py            # QNetwork, ring replay buffer, Double DQN step
│       ├── planner.py       # greedy expected-value planner (+ DQN tie-break)
│       ├── train.py         # trains the JEPA          (checkpoints/latest.pt)
│       ├── train_rl.py      # trains the DQN           (checkpoints/rl_v2*.pt)
│       ├── eval_rl.py       # benchmarks + baselines + calibration report
│       └── play.py          # interactive viewer
│
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python >= 3.11
- PyTorch, Torchvision, tqdm, matplotlib
- CUDA supported when a compatible NVIDIA environment is available

The project uses **uv** for environment and dependency management.

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

Produces `checkpoints/latest.pt`.

> Any change to `model.py`, `data.py` or the loss makes the previous checkpoint stale. **Checkpoints from before the AccuracyHead cannot be loaded (missing `accuracy_head.*` keys) and must be regenerated.**

### Train DQN

```bash
uv run train-rl
```

Produces `checkpoints/rl_v2.pt` and `rl_v2_best.pt`.

> Re-run whenever the JEPA checkpoint changes or the reward/exploration hyperparameters change. The *meaning* of the 168-dim state (especially the candidate features) changes with every retrained JEPA, so an old DQN checkpoint is not valid for a new one.

### Evaluate

```bash
uv run eval-rl
```

(or add `eval-rl = "active_jepa.eval_rl:main"` to `[project.scripts]`).

---

## Running the Agent

```bash
uv run play-jepa
```

---

## Why Hierarchical JEPA + RL?

**Local / mid JEPA levels** learn *"what will this region look like once revealed?"*; **the global level** learns *"what digit is this, given everything seen so far?"*; **the AccuracyHead** learns *"how much does looking here help?"*; **RL** learns *"what should I look at next, and when should I stop?"* This separates **representation learning** from **decision making**.

---

## Future Work

- Compare against the `fixed@t` / `random@t` baselines across reveal costs (accuracy-vs-cells curve)
- Re-fit the AccuracyHead on held-out images after JEPA training (it is currently trained on the JEPA split, where the classifier is slightly over-confident)
- Multi-step JEPA prediction / lookahead beyond one reveal
- Vectorized environments; cached JEPA representations
- Vision Transformer encoder, Transformer predictor
- Actor-Critic instead of DQN
- Curiosity / intrinsic reward
- Larger images and more complex environments than MNIST

---

## Status

An experimental research/learning project exploring hierarchical JEPA, active perception, value-of-information estimation and DQN on a deliberately simple environment (MNIST + 3×3 region selection). The interesting question is **whether an agent can learn when and where to look** — `eval_rl.py` is how you find out.
