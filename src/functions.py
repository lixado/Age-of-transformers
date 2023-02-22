import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm


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
