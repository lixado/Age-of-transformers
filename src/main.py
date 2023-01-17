import torch
import matplotlib
import time

from logger import Logger

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())

print(time.strftime("%Y%m%d-%H-%M-%S"))

logger = Logger()

import random
import numpy

import random
from DeepRTS import python
from DeepRTS import Engine

from DeepRTS.python import scenario

if __name__ == "__main__":

    episodes = 10000000
    random_play = True
    gui_config = python.Config(
        render=True,
        view=True,
        inputs=True,
        caption=False,
        unit_health=True,
        unit_outline=False,
        unit_animation=True,
        audio=False,
        audio_volume=50
    )

    engine_config: Engine.Config = Engine.Config.defaults()
    engine_config.set_barracks(True)


    env = scenario.GoldCollectOnePlayerFifteen({})
    game = env.game


    game.set_max_fps(30)
    game.set_max_ups(10000000)

    for episode in range(episodes):
        print("Episode: %s, FPS: %s, UPS: %s" % (episode, game.get_fps(), game.get_ups()))

        terminal = False
        state = env.reset()
        while not terminal:
            action = random.randint(0, 15)  # TODO AI Goes here
            next_state, reward, terminal, _ = env.step(action)

            state = next_state