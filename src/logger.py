import os
import time
import shutil


class Logger():
    def __init__(self):
        # create folder for results
        self.folder_path = "./results/" + time.strftime("%Y%m%d-%H-%M-%S")
        os.mkdir(self.folder_path)

        # copy configs
        shutil.copy2("./src/config.json", self.folder_path)

        print("Saved to folder " + os.path.abspath(self.folder_path)) 


    def save_epoch(self):
        pass

    def save_model(self):
        pass