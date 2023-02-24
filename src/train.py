import os
from functions import SaveTempImage, CreateVideoFromTempImages

def train(env, agent, logger, config):
    agent.net.train()

    record_epochs = 5  # record game every x epochs
    for e in range(config["epochs"]):
        observation, info = env.reset()
        ticks = 0
        recording_images = []
        record = (e % record_epochs == 0) or (e == config["epochs"] - 1)  # last part to always record last
        if record:
            print("Recording this epoch")
        done = False
        while not done and ticks < 8000:
            ticks += 1

            # Record game
            if record:
                SaveTempImage(logger.getSaveFolderPath(), env.render(), ticks)

            # AI choose action
            actionIndex = agent.act(observation)

            # Act
            next_observation, reward, done, _, info = env.step(actionIndex)

            # use this to see image example
            # cv2.imshow('image', next_observation[0])
            # cv2.waitKey(3)

            # AI Save memory
            agent.cache(observation, next_observation, actionIndex, reward, done)

            # Learn
            q, loss = agent.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            observation = next_observation

        logger.log_epoch(e, agent.exploration_rate)

        # Record game
        if record:
            CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e + 1))

    # save model
    agent.save()
    env.close()