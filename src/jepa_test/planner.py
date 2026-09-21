"""
Hybrid planner -- greedy expected-value planner + optional DQN tie-breaker.

Shared by play.py (interactive) and eval_rl.py (benchmarks), so it has no
matplotlib dependency.

For every step it computes an expected value for every possible action using
the same economics as train_rl.py:

    value(STOP)      = P(correct now)        * correct_reward
                     + (1 - P(correct now))  * wrong_reward
    value(reveal c)  = -reveal_cost
                     + P(correct after c)       * correct_reward
                     + (1 - P(correct after c)) * wrong_reward

where the probabilities come from the JEPA's AccuracyHead (trained against
the classifier's correctness on the REAL next observation). Previously they
were the confidence of the classifier applied to an imagined z_pred, which is
not an estimate of the value of information.

If exactly one action is (approximately) the best, that's the move. If
several are within `tie_threshold` of the best (a tie the one-step lookahead
can't resolve), the DQN's Q-values pick among ONLY those tied actions. With
`q_network=None` the planner is purely greedy (argmax of value).
"""

import torch

from .data import NUM_CELLS, STOP_ACTION
from .environment import build_state

REVEAL_COST = 0.20
CORRECT_REWARD = 1.0
WRONG_REWARD = -1.0

# How close two actions' expected values need to be to count as "tied".
TIE_THRESHOLD = 0.03


def expected_value(
    p_correct,
    correct_reward=CORRECT_REWARD,
    wrong_reward=WRONG_REWARD,
):
    return p_correct * correct_reward + (1.0 - p_correct) * wrong_reward


@torch.no_grad()
def plan_action(
    model,
    q_network,
    image,
    opened,
    device,
    reveal_cost=REVEAL_COST,
    correct_reward=CORRECT_REWARD,
    wrong_reward=WRONG_REWARD,
    tie_threshold=TIE_THRESHOLD,
):
    """
    Returns a dict:
        action             chosen action (0-8 reveal, 9 STOP)
        values             {action: expected value}
        accuracy           {action: predicted P(correct)}  (STOP -> now)
        tied_actions       actions within `tie_threshold` of the best value
        tie_broken_by_dqn  True if the DQN made the final choice
        state, info        from build_state (info has z/tokens/probabilities)
    """

    state, info = build_state(
        model,
        image,
        opened,
        device,
    )

    acc_now = info["acc_now"].item()

    acc_after = info["acc_after"].tolist()

    accuracy = {STOP_ACTION: acc_now}

    values = {
        STOP_ACTION: expected_value(
            acc_now,
            correct_reward,
            wrong_reward,
        )
    }

    for cell in range(NUM_CELLS):
        if cell in opened:
            continue

        accuracy[cell] = acc_after[cell]

        values[cell] = -reveal_cost + expected_value(
            acc_after[cell],
            correct_reward,
            wrong_reward,
        )

    best_value = max(values.values())

    tied_actions = [
        action
        for action, value in values.items()
        if best_value - value <= tie_threshold
    ]

    tie_broken_by_dqn = False

    if q_network is None:
        # Purely greedy.
        action = max(
            values,
            key=values.get,
        )

    elif len(tied_actions) == 1:
        action = tied_actions[0]

    else:
        # Genuine tie the greedy one-step lookahead can't resolve on its own
        # -> ask the DQN, but restrict its choice to ONLY the tied actions
        # (never let it pick something already ruled out as clearly worse).
        tie_broken_by_dqn = True

        q_values = q_network(state.unsqueeze(0)).squeeze(0)

        masked_q = torch.full_like(
            q_values,
            float("-inf"),
        )

        for tied_action in tied_actions:
            masked_q[tied_action] = q_values[tied_action]

        action = masked_q.argmax().item()

    return {
        "action": action,
        "values": values,
        "accuracy": accuracy,
        "tied_actions": tied_actions,
        "tie_broken_by_dqn": tie_broken_by_dqn,
        "state": state,
        "info": info,
    }
