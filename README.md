# Age-of-transformers
Transformers based AI to learn to play deep-rts

# Setup
1.     Create enviroment and download dependencies
 - `python -m venv env`
 - `source env/bin/activate && pip install -r requirements.txt` 

2.     Download and compile deep-rts
 - `git submodule init && git submodule update`
 - `cd deep-rts && git submodule init && git submodule update`
 - `sudo apt install ccache`
 - `pip install .`


# Extra commands

- To save all dependencies: `pip freeze > requirements.txt`
- Remove all pip packages: `pip freeze | xargs pip uninstall -y` if error `pip uninstall DeepRTS`