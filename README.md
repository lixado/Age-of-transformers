# Age-of-transformers
Transformers based AI to learn to play deep-rts

# Setup
1.     Create enviroment and download dependencies
 - `python -m venv env`
 - `source env/bin/activate && pip install -r requirements.txt` 

2.     Download and compile deep-rts
 - `git submodule init && git submodule update`
 - `cd deep-rts && git submodule init && git submodule update`
 - `sudo apt install ccache libgtk-3-dev cmake`
 - `pip install .`

# Docker
 - `cd age-of-transformers`
 - `docker build --tag age-of-transformers .`
 - `docker run -it age-of-transformers`
 - When the container has started run: `pip install deep-rts/`
   - If this command fails after some time, try to run the command again and it should install successfully (DeepRTS should show up after running `pip list`)


# Extra commands

- To save all dependencies: `pip freeze > requirements.txt`
- Remove all pip packages: `pip freeze | xargs pip uninstall -y` if error `pip uninstall DeepRTS`
- To update submodules: `git submodule sync`
