import os
import time
import shutil
import datetime
import matplotlib
matplotlib.use("agg") # dix main thread bug https://github.com/matplotlib/matplotlib/issues/23419#issuecomment-1182530883
import matplotlib.pyplot as plt
import numpy as np

class Logger():
    def __init__(self, workingDir):
        # create folder for results
        self.folder_path = os.path.join(workingDir, "results", time.strftime("%Y%m%d-%H-%M-%S"))
        os.makedirs(self.folder_path, exist_ok=True)
        self.movingAvgNumber = 100 # take last 100 and average for plotting

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
        self.movingAvgrewards = []
        self.movingAvglosses = []
        self.movingAvgactions = []
        self.movingAvgqs = []
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

    def log_GADT(self, loss):
        path = os.path.join(self.folder_path, "GADT_losses.jpg")

        if not hasattr(self, 'loss_GADT'):
            self.loss_GADT = []
        self.loss_GADT.append(loss)

        if not hasattr(self, 'movingavg_loss_GADT'):
            self.movingavg_loss_GADT = []
        self.movingavg_loss_GADT.append(np.round(np.mean(self.loss_GADT[-self.movingAvgNumber:]), 4))


        plt.plot(self.movingavg_loss_GADT)
        plt.title(f"Avg of previous {self.movingAvgNumber} epochs of sum of loss per epoch")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.savefig(path)
        plt.clf()


    def log_epoch(self, epoch, epsilon, lr, print_console=True):
        tNow = time.time()
        tDelta = tNow - self.record_time

        avgReward = np.round(self.epochTotalReward / self.epochTotalActions, 5)
        avgQ = np.round(self.epochTotalQ / self.epochTotalActions, 5)
        avgLoss = np.round(self.epochTotalLoss / self.epochTotalActions, 5)

        self.rewards.append(self.epochTotalReward)
        self.losses.append(self.epochTotalLoss)
        self.actions.append(self.epochTotalActions)
        self.qs.append(self.epochTotalQ)

        self.movingAvgrewards.append(np.round(np.mean(self.rewards[-self.movingAvgNumber:]), 4))
        self.movingAvglosses.append(np.round(np.mean(self.losses[-self.movingAvgNumber:]), 4))
        self.movingAvgactions.append(np.round(np.mean(self.actions[-self.movingAvgNumber:]), 4))
        self.movingAvgqs.append(np.round(np.mean(self.qs[-self.movingAvgNumber:]), 4))

        if print_console:
            print(
                f"Epoch {epoch} - "
                f"Actions this epoch {self.epochTotalActions} - "
                f"Epsilon {epsilon} - "
                f"Lr {lr:.5f} - "
                f"Avg reward {avgReward} - "
                f"Avg loss {avgLoss} - "
                f"Avg Q {avgQ} - "
                f"Time Delta {tDelta} - "
                f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
            )

        with open(self.save_log, "a") as f:
            f.write(
                f"{epoch:8d}{self.epochTotalActions:8d}{epsilon:10.3f}"
                f"{(np.mean(self.rewards[-self.movingAvgNumber:])):15.3f}{(np.mean(self.losses[-self.movingAvgNumber:])):15.3f}{(np.mean(self.qs[-self.movingAvgNumber:])):15.3f}"
                f"{tDelta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
            
        for metric in ["rewards", "losses", "actions", "qs"]:
            plt.plot(getattr(self, f"movingAvg{metric}"))
            plt.title(f"Avg of previous {self.movingAvgNumber} epochs of sum of {metric} per epoch")
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

        self.initEpochVariables()

    def getSaveFolderPath(self):
        return self.folder_path
