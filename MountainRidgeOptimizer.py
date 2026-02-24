from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from MountainRidge import MountainRidge
from Agent import Agent, DTYPE

class MountainRidgeOptimizer:
    def __init__(self, h: int, w: int, seed=4212, agent_num=5):
        self.space = MountainRidge(h, w, seed)
        Agent.set_space(self.space)
        self.agents = [Agent(seed + i) for i in range(agent_num)]

    def get_search_space(self) -> NDArray:
        """Get the height map of the search space"""
        return self.space.get_height_map()

    def get_swarm_positions(self) -> List[Tuple[DTYPE, DTYPE]]:
        """Get a list of tuples representing the (x,y) coordinates of every agent in this iteration"""
        return [agent.pos for agent in self.agents]

    def step(self, greed: float = 1, social: float = 1, chaos: float = 0.5) -> None:
        """Perform a single iteration.
        Each agent moves, and after all agents moved they will all check if they're the best"""
        [agent.move(greed, social, chaos) for agent in self.agents]
        Agent.best_agent = None
        [agent.check_if_best() for agent in self.agents]

    @staticmethod
    def get_best() -> Tuple[Tuple[DTYPE, DTYPE], float]:
        """Get coordinates and height of current best in the population
        The output is a Tuple of two elements, which are:
        1. a Tuple of DTYPE as defined in Agent, representing the coordinates of the best agent
        2. The height at the position of the best agent"""
        return Agent.best_agent.pos, Agent.best_score

    def get_global_minimum(self) -> Tuple[Tuple[DTYPE, DTYPE], float]:
        """Get coordinates and height of global best"""
        height_map = self.space.get_height_map()
        linear_index = np.argmin(height_map)
        coordinates = np.unravel_index(linear_index, height_map.shape)
        coordinates = (coordinates[0].astype(DTYPE), coordinates[1].astype(DTYPE))
        minimum_height = float(height_map[coordinates])
        return coordinates, minimum_height
