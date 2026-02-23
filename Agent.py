from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy import random
from numpy.typing import NDArray
from MountainRidge import MountainRidge

DTYPE = np.uint16
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
        if Agent.best_score > self.score:
            Agent.best_agent, Agent.best_score = self, self.score
            return True
        return False

    def calc_move_value(self, neighborhood: NDArray, direc: Tuple[int, int],
                        greed: float, social: float, chaos: float) -> float:
        # TODO: can be replaced with something neatly vectorized
        """
        Calculate the value of a move
        :param neighborhood: Cell's Moore neighborhood
        :param direc: direction of movement
        :param greed: weight of greedy factor
        :param social: weight of social factor
        :param chaos: weight of random movement
        :return: move value (would need to be normalized if I intend to use it as probabilities)
        """
        # Do not move out of bounds
        x, y = self.pos
        if x + direc[0] < 0 or y + direc[1] < 0:
            return 0

        # add greed value, which is the difference between current score and score in this direction
        value = (neighborhood[direc] - self.score) * greed
        # add social component - difference between current score and global best IF this direction gets the agent
        # closer to this global best
        np_pos = np.array(self.pos, dtype=DTYPE)
        best_pos = np.array(Agent.best_agent.get_pos(), dtype=DTYPE)
        curr_distance = np.linalg.norm(np_pos - best_pos)
        new_distance = np.linalg.norm(np_pos + np.array(direc, dtype=DTYPE) - best_pos)
        if new_distance < curr_distance:
            value += (neighborhood[direc] - self.score) * social
        # add noise
        value += self.rng.random() * chaos

        return value

    def move(self, greed: float = 1, social: float = 1, chaos: float = 0.5) -> Tuple[DTYPE, DTYPE]:
        """
        Choose agent's next move according to immediate neighborhood, the social factor and some randomness.
        :param greed: weight of greedy factor
        :param social: weight of social factor
        :param chaos: weight of randomness (random values are in the range [0,1])
        :return: New coordinates after move
        """
        neighborhood = Agent.space.get_neighborhood(self.pos[0], self.pos[1])
        move_values = [self.calc_move_value(neighborhood, direction, greed, social, chaos) for direction in DIRS]
        value_sum = sum(move_values)
        normalized_values = [val / value_sum for val in move_values]
        return self.rng.choice(DIRS, p=normalized_values)

