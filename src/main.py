import random
from DeepRTS.python import Config
from DeepRTS.python import scenario

if __name__ == "__main__":
    random_play = True
    episodes = 100

    for i in range(episodes):
        env = scenario.GeneralAI_1v1(Config.Map.THIRTYONE)
        state = env.reset()
        done = False

        while not done:
            env.game.set_player(env.game.players[0])
            action = random.randrange(15)
            next_state, reward, done, _ = env.step(action)
            state = next_state

            if (done):
                break

            env.game.set_player(env.game.players[1])
            action = random.randrange(15)
            next_state, reward, done, _ = env.step(action)
            state = next_state
