from MountainRidgeOptimizer import MountainRidgeOptimizer
import gui
from math import floor

width = 2048
aspect_ratio = 3/4
height = floor(width * aspect_ratio)
# height, width = 512, 512
seed = 6666
agent_num = 20

greed = 1.0
social = 0.5
chaos = 0.25
update_in_ms = 20

mountain_ridge = MountainRidgeOptimizer(height, width, seed, agent_num)

if __name__ == '__main__':
    gui.run(mountain_ridge, greed, social, chaos, update_in_ms)