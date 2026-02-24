from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy import random
from numpy.typing import NDArray
from MountainRidge import MountainRidge

DTYPE = np.int16
DIRS = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1, -1), (0, -1), (1, -1)]   # possible move directions
ETA = 1e-6  # small constant

class Agent:
    # search space used by all agents
    space: MountainRidge|None = None

    @staticmethod
    def set_space(space: MountainRidge):
        Agent.space = space

    # global memory
    best_agent: Agent|None = None
    best_score: float|None = None

    # Individual Agents
    def __init__(self, seed: int):
        space_height, space_width = Agent.space.shape()
        self.rng = random.default_rng(seed)
        self.pos: Tuple[DTYPE, DTYPE] = (self.rng.integers(0, space_height, dtype= DTYPE),
                                         self.rng.integers(0, space_width, dtype= DTYPE))
        self.score = Agent.space.get_height(self.pos[0], self.pos[1])

        if Agent.best_agent is None:
            Agent.best_agent = self
            Agent.best_score = self.score

    def get_pos(self) -> Tuple[DTYPE, DTYPE]:
        return self.pos

    def check_if_best(self) -> bool:
        if Agent.best_agent is None or Agent.best_score > self.score:
            Agent.best_agent, Agent.best_score = self, self.score
            return True
        return False

    def move(self, greed: float, social: float, chaos: float) -> Tuple[int, int]:
        """
        Choose a neighboring cell (Moore neighborhood) at random with probability
        proportional to a value computed per-neighbor.

        Value for a candidate neighbor (nx, ny) is:
            value = greedy_contrib + social_contrib + random_contrib

        - greedy_contrib = greed * max(0, current_height - neighbor_height)
            (we are minimizing height, so larger improvement => larger contribution)
        - social_contrib = social * max(0, curr_dist_to_best - new_dist_to_best)
            (only if the move reduces distance to current global best; proportional
             to how much closer it gets)
        - random_contrib = chaos * rng.random()  (uniform in [0,1) scaled by chaos)

        Cells outside the map get value == 0 and are never selected.
        If no neighbor has positive value, the agent does not move (returns (0,0)).
        Returns the chosen direction tuple (dx, dy) — same contract as original code.
        """
        # defensive casts to plain Python ints (avoid numpy unsigned overflow issues)
        x = int(self.pos[0])
        y = int(self.pos[1])

        # get heightmap for bounds checking and (fast) height lookups
        height_map = Agent.space.get_height_map()
        rows, cols = height_map.shape

        # current height / score and current distance to global best (if defined)
        curr_height = float(self.score)
        if Agent.best_agent is not None:
            best_pos = Agent.best_agent.get_pos()
            best_x = int(best_pos[0])
            best_y = int(best_pos[1])
            # current euclidean distance to best
            curr_dist = np.hypot(x - best_x, y - best_y)
        else:
            # no social information available
            best_x = best_y = None
            curr_dist = None

        values = []
        candidate_dirs = []

        for dx, dy in DIRS:
            nx = x + dx
            ny = y + dy

            # out-of-bounds neighbors get 0 value
            if nx < 0 or ny < 0 or nx >= rows or ny >= cols:
                # explicitly record zero-valued neighbor (but we won't select it)
                values.append(0.0)
                candidate_dirs.append((dx, dy))
                continue

            # neighbor height (float)
            neigh_h = float(Agent.space.get_height(nx, ny))

            # greedy: improvement in height (positive if neighbor is lower)
            greedy_contrib = greed * max(0.0, curr_height - neigh_h)

            # social: only if we have a best agent and the move reduces distance to it
            social_contrib = 0.0
            if curr_dist is not None:
                new_dist = float(np.hypot(nx - best_x, ny - best_y))
                if new_dist < curr_dist:
                    # contribution proportional to how much closer we get
                    social_contrib = social * (curr_dist - new_dist)

            # random noise
            random_contrib = chaos * float(self.rng.random())

            # total (force non-negative)
            total = greedy_contrib + social_contrib + random_contrib
            if total < 0.0:
                total = 0.0

            values.append(float(total))
            candidate_dirs.append((dx, dy))

        # Build list of strictly positive candidates and their weights
        pos_dirs_and_weights = [(d, w) for d, w in zip(candidate_dirs, values) if w > 0.0]

        if not pos_dirs_and_weights:
            # no positive-valued move -> stay in place (return a zero offset)
            return (0, 0)

        dirs, weights = zip(*pos_dirs_and_weights)
        weights = np.array(weights, dtype=float)
        weights_sum = weights.sum()
        if weights_sum <= 0.0:
            # numerical safety: fallback to uniform choice among dirs
            chosen_dir = tuple(self.rng.choice(len(dirs)))
            # but above line isn't useful; better choose uniformly:
            chosen_dir = tuple(dirs[self.rng.integers(0, len(dirs))])
        else:
            probs = weights / weights_sum
            idx = self.rng.choice(len(dirs), p=probs)
            chosen_dir = dirs[idx]

        # update position and score
        new_x = x + chosen_dir[0]
        new_y = y + chosen_dir[1]

        # final safety check (shouldn't be necessary given above checks)
        if new_x < 0 or new_y < 0 or new_x >= rows or new_y >= cols:
            # do not move if something went wrong
            return (0, 0)

        self.pos = (DTYPE(new_x), DTYPE(new_y))
        self.score = float(Agent.space.get_height(new_x, new_y))

        # Update global best if needed
        self.check_if_best()

