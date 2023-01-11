# Age-of-transformers
Transformers based AI to learn to play openage

# Setup
1.     Create enviroment and download requirements
 - `python -m venv env`
 - `source env/bin/activate && pip install -r requirements.txt` 

2.    Download deep-rts
 - `git submodule init`
 - `git submodule update` 

3.    Compile deep-rts
 - `cd deep-rts && git submodule sync && git submodule update --init && pip install .`


# Extra commands

### To save all dependencies: `pip freeze > requirements.txt`
