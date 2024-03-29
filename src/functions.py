import os
import json
import random
import re
import shutil
import sys
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
        self.score = player.get_score()
        self.statistic_damage_done = player.statistic_damage_done
        self.statistic_damage_taken = player.statistic_damage_taken
        self.statistic_gathered_gold = player.statistic_gathered_gold
        self.statistic_gathered_lumber = player.statistic_gathered_lumber
        self.statistic_gathered_stone = player.statistic_gathered_stone
        self.statistic_units_created = player.statistic_units_created
        self.num_town_hall = player.num_town_hall
        self.num_barrack = player.num_barrack
        self.num_farm = player.num_farm
        self.num_peasant = player.num_peasant
        self.num_footman = player.num_footman

    def evaluate_player_state(self):
        return self.player_state

    def get_score(self):
        return self.score
    

def chooseModel(folderPath):
    folderList = [name for name in os.listdir(folderPath) if os.path.isdir(os.path.join(folderPath, name)) and len(os.listdir(os.path.join(folderPath, name))) != 0]

    if len(folderList) == 0:
        print("No models to load in path: ", folderPath)
        quit()

    for cnt, fileName in enumerate(folderList, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

    choice = int(input("Select folder with platformer model[1-%s]: " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(os.path.join(folderPath, folder)) if f.endswith(".chkpt")]

    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    modelPath = os.path.join(folderPath, folder, fileList[0])
    return modelPath

def NotifyDiscord(message):
    data = {
        "content": message
    }
    url = "https://discord.com/api/webhooks/1076092503238922290/Rtdbr-HBf7O2mzAUwz95xW8Qjrgp12bloT0ygA6qICtoA9uwozY4X4DzYEMGPJLKUE91"
    result = requests.post(url, json=data)

def GetConfigDict(workingDirPath) -> dict:
    configPath = os.path.join(workingDirPath, "src", "config.json")
    try:
        with open(configPath) as json_file:
            data = json.load(json_file)
            return data
    except:
        print(f'Could not read config file in {configPath}')


def CreateVideoFromTempImages(images_folder, epoch):
    images = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(".png")]
    # order arrays
    def sortFunc(imgPath) -> int:
            basepath = str(os.path.basename(imgPath))
            nr = basepath.split(".")[0] # remove extention
            return int(nr)
    
    images.sort(key=sortFunc)

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    log_folder = os.path.dirname(images_folder) # get parent dir
    path = os.path.join(log_folder, f"video{epoch}.mp4")
    video = cv2.VideoWriter(path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=1, frameSize=(width, height))

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
    path_to_image = os.path.join(path_to_image, f'{number}.png')
    cv2.imwrite(path_to_image, image)
    
def NotifyDiscord(message):
    data = {
        "content": message
    }
    url = "https://discord.com/api/webhooks/1076092503238922290/Rtdbr-HBf7O2mzAUwz95xW8Qjrgp12bloT0ygA6qICtoA9uwozY4X4DzYEMGPJLKUE91"
    result = requests.post(url, json=data)

def sample_with_order(population, max_sequence_length):
    """
        Samples [1,population] elements from a sequence mantaining the order
    """
    k = random.randint(1, max_sequence_length)
    k_start = random.randint(0, len(population)-k)
    return population[k_start: k_start+k]

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]