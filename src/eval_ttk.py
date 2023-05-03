import os
import sys
import gym
import pygame
import time


def evaluate(agent, gym: gym.Env, modelPath):
    workingDir = os.getcwd()
    if not os.path.exists(os.path.join(workingDir, "src")):
        sys.exit(
            f'Working directory: {workingDir} not correct, should be "Age-of-transformers/" not "{os.path.basename(os.path.normpath(workingDir))}"')

    #logger = Logger(workingDir) might be needed

    pygame.init()
    pygame.display.set_caption('DeepRTS v3.0')  # set the pygame window name

    WIDTH = gym.initial_shape[0]*2 # because of dashboard
    HEIGHT = gym.initial_shape[1]
    canvas = pygame.display.set_mode((WIDTH, HEIGHT))

    #Set agent model and evaluation mode
    agent.loadModel(modelPath)
    agent.net.eval()

    play = True
    while play:
        state, info = gym.reset()
        done = False
        j = 0
        start = time.time()
        while True:#not done:
            j += 1
            #Actions
            action0, q_values = agent.act(state)
            state, reward, done, _, _ = gym.step(action0)


            image = gym.render(q_values)

            img = pygame.transform.rotate(pygame.surfarray.make_surface(image), -90)
            img = pygame.transform.flip(img, True, False)

            canvas.blit(img, (0, 0))
            pygame.display.update()


            ev = pygame.event.get()
            action1 = action(ev)

            if action1 == -1: # pressed escape leave
                play = False
                break

            gym.player1.do_action(action1)

            if done:
                break
            
        print("Done")
    gym.close()
    pygame.quit()

