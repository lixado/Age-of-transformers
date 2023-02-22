import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import requests
from DeepRTS import Engine, Constants

class PlayerState():
    def __init__(self, player):
        # Current states
        self.player_state = player.evaluate_player_state()  # Defeat, victory or playing
        self.gold = player.gold
        self.lumber = player.lumber
        self.stone = player.stone
        self.food = player.food

        # Statistics
        self.statistic_damage_done = player.statistic_damage_done
        self.statistic_damage_taken = player.statistic_damage_taken
        self.statistic_gathered_gold = player.statistic_gathered_gold
        self.statistic_gathered_lumber = player.statistic_gathered_lumber
        self.statistic_gathered_stone = player.statistic_gathered_stone
        self.statistic_units_created = player.statistic_units_created

    def evaluate_player_state(self):
        return self.player_state

def get_reward(mode, player0, previousPlayer0, player1, steps):
    if mode==0:
        return calc_reward(player0, player1, steps)
    if mode==1:
        return conditional_reward(player0, previousPlayer0, player1)
    return 0

def calc_reward(player0, player1, steps):
    winnerReward = int(player1.evaluate_player_state() == Constants.PlayerState.Defeat) * 10 - int(
        player0.evaluate_player_state() == Constants.PlayerState.Defeat) * 10  # if win then +10 if loss -10
    dmgReward = 1 - ((100 - player0.statistic_damage_done) / 100) ** 0.5  # exponentially based on dmg done, 100=max dmg
    timeConservation = (8000 - steps) / 8000  # * the dmg reward, higher the lesser time has passed
    return dmgReward * timeConservation + winnerReward

def conditional_reward(player0, previousPlayer0: PlayerState, player1):
    if player0.evaluate_player_state() != Constants.PlayerState.Defeat and player1.evaluate_player_state() == Constants.PlayerState.Defeat:
        return 1000
    if player0.evaluate_player_state() == Constants.PlayerState.Defeat and player1.evaluate_player_state() != Constants.PlayerState.Defeat:
        return -100
    if player0.statistic_damage_done > previousPlayer0.statistic_damage_done and player1.statistic_damage_taken > 0:
        return 100
    return 0

def GetConfigDict(workingDirPath):
    configPath = os.path.join(workingDirPath, "src", "config.json")
    try:
        with open(configPath) as json_file:
            data = json.load(json_file)
            return data
    except:
        print(f'Could not read config file in {configPath}')
        

def TransformImage(image, image_shape):
    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resize = cv2.resize(gray, image_shape[::-1])
    framestack = np.expand_dims(resize, 0) # fake framestack for neural network dimentions
    
    return framestack


def CreateVideoFromTempImages(images_folder, epoch):
    images = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(".png")]

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    log_folder = os.path.dirname(images_folder) # get parent dir
    path = os.path.join(log_folder, f"video{epoch}.avi")
    video = cv2.VideoWriter(path, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=240, frameSize=(width, height))

    for image in tqdm(images, desc="Processing images"):
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    # NEED TO DELETE TEMP folder and all images
    shutil.rmtree(images_folder)

    print("Saved video to path: ", path)


def SaveTempImage(resultsFolder, image, number):
    path_to_image = os.path.join(resultsFolder, "temp")
    os.makedirs(path_to_image, exist_ok=True) # create temp folder if not exist
    path_to_image = os.path.join(path_to_image, f'image{number}.png')
    cv2.imwrite(path_to_image, image)
    
def NotifyDiscord(message):
    data = {
        "content": message
    }
    url = "https://discord.com/api/webhooks/1076092503238922290/Rtdbr-HBf7O2mzAUwz95xW8Qjrgp12bloT0ygA6qICtoA9uwozY4X4DzYEMGPJLKUE91"
    result = requests.post(url, json=data)
