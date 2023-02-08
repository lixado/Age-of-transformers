import os
import time
import shutil


class Logger():
    def __init__(self, workingDir):
        # create folder for results
        self.folder_path = os.path.join(workingDir, "results", time.strftime("%Y%m%d-%H-%M-%S"))
        os.makedirs(self.folder_path, exist_ok=True)

        # copy configs
        shutil.copy2("./src/config.json", self.folder_path)

        print("Saved to folder " + os.path.abspath(self.folder_path)) 

    def getSaveSolver(self):
        return self.folder_path

    def save_epoch(self):
        pass

    def save_model(self):
        pass