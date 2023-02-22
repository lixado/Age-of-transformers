import os
import time
import shutil
import datetime
import matplotlib
matplotlib.use("agg") # dix main thread bug https://github.com/matplotlib/matplotlib/issues/23419#issuecomment-1182530883
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, workingDir):
        # create folder for results
        self.folder_path = os.path.join(workingDir, "results", time.strftime("%Y%m%d-%H-%M-%S"))
        os.makedirs(self.folder_path, exist_ok=True)

        # copy configs
        shutil.copy2(os.path.join(workingDir, "src", "config.json"), self.folder_path)

        # create log file
        self.save_log = os.path.join(self.folder_path, "log.txt")
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Steps':>8}{'Epsilon':>10}{'Reward':>15}"
                f"{'Loss':>15}{'Q':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        print("Saved to folder " + self.folder_path)

        # initiate variables
        self.rewards = []
        self.losses = []
        self.actions = []
        self.qs = []
        self.rewards_plot = os.path.join(self.folder_path, "rewards.jpg")
        self.losses_plot = os.path.join(self.folder_path, "losses.jpg")
        self.actions_plot = os.path.join(self.folder_path, "actions.jpg")
        self.qs_plot = os.path.join(self.folder_path, "qs.jpg")
        self.initEpochVariables()

    def initEpochVariables(self):
        self.epochTotalReward = 0
        self.epochTotalLoss = 0
        self.epochTotalActions = 0
        self.epochTotalQ = 0
        self.record_time = time.time()

    def log_step(self, reward, loss , q):
        self.epochTotalReward += reward
        self.epochTotalLoss += loss
        self.epochTotalActions += 1
        self.epochTotalQ += q

    def log_epoch(self, epoch, epsilon):
        tNow = time.time()
        tDelta = tNow - self.record_time

        self.rewards.append(self.epochTotalReward)
        self.losses.append(self.epochTotalLoss)
        self.actions.append(self.epochTotalActions)
        self.qs.append(self.epochTotalQ)

        print(
            f"Epoch {epoch} - "
            f"Actions this epoch {self.epochTotalActions} - "
            f"Epsilon {epsilon} - "
            f"Reward this epoch {self.epochTotalReward} - "
            f"Loss this epoch {self.epochTotalLoss} - "
            f"Q Value this epoch {self.epochTotalQ} - "
            f"Time Delta {tDelta} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{epoch:8d}{self.epochTotalActions:8d}{epsilon:10.3f}"
                f"{self.epochTotalReward:15.3f}{self.epochTotalLoss:15.3f}{self.epochTotalQ:15.3f}"
                f"{tDelta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
            
        for metric in ["rewards", "losses", "actions", "qs"]:
            plt.plot(getattr(self, metric))
            plt.title(f"Sum of {metric} this epoch")
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

        self.initEpochVariables()

    def getSaveFolderPath(self):
        return self.folder_path