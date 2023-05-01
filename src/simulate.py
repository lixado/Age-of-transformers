import random
import pickle
import time

import gym
import os
from functions import CreateVideoFromTempImages, SaveTempImage
import pygame
from typing import List


def simulate(config: dict, agent, gym: gym.Env, logger, model_path=None):
    pygame.init()
    pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name
    size = (gym.render_shape[0]*2, gym.render_shape[1])
    canvas = pygame.display.set_mode(size)  # pygame.FULLSCREEN

    """
        Start engine
    """
    gym.game.set_max_fps(60)  # 0 = unlimited


    record_epochs = config["recordEvery"] # record game every x epochs
    epochs = config["epochs"]
    save_path = logger.getSaveFolderPath()
    data_path = os.path.join(save_path, "data")
    os.makedirs(data_path, exist_ok=True) # create temp folder if not exist

    if model_path != None:
        agent.loadModel(model_path)
        agent.net.eval()
    for e in range(3, epochs-3):
        observation, info = gym.reset()
        ticks = 0
        record = (e % record_epochs == 0) or (e == epochs-1) # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        truncated = False
        memory = []
        while not done and not truncated:
            ticks += 1

            # event loop
            ev = pygame.event.get()
            action = Controls(ev)  # default do nothing
            if action == -1:
                done = True
                break

            if model_path != None:
                actionIndex, q_values = agent.act(observation)
            else:
                actionIndex, q_values = action, [0] * len(gym.action_space)

            gym.save_player_state()

            # Act
            next_observation, reward, done, truncated, info = gym.step(actionIndex)

            image = gym.render(q_values)

            img = pygame.transform.rotate(pygame.surfarray.make_surface(image), -90)
            img = pygame.transform.flip(img, True, False)

            canvas.blit(img, (0, 0))
            pygame.display.update()


            # Record game
            if record:
                SaveTempImage(save_path, image, ticks)

            logger.log_step(reward, 0, 0)

            # save data
            memory.append([observation[-1], actionIndex, ticks, reward])
            # Update state
            observation = next_observation
            time.sleep(1)

        logger.log_epoch(e, agent.exploration_rate, 0)

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(save_path, "temp"), (e))
        print(f"Epoch {e} done")

        # save game data
        with open(f"{os.path.join(data_path, f'game_{e:04}.pickle')}", "wb") as f:
            pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
    gym.close()

def Controls(ev: List[pygame.event.Event]):
    for event in ev:
        if event.type == pygame.KEYDOWN: # if some key was pressed
            match event.key:
                case pygame.K_1:
                    action = 0
                case pygame.K_2:
                    action = 1

                case pygame.K_UP:
                    action = 2
                case pygame.K_DOWN:
                    action = 3
                case pygame.K_LEFT:
                    action = 4
                case pygame.K_RIGHT:
                    action = 5

                case pygame.K_SPACE:
                    action = 10
                case pygame.K_x:
                    action = 11

                case pygame.K_3:
                    action = 12
                case pygame.K_4:
                    action = 13
                case pygame.K_5:
                    action = 14
            return action
        if event.type == pygame.KEYUP:
            match event.key:
                case pygame.K_ESCAPE:
                    action = -1
    return 15 # default
