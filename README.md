# Age-of-transformers
Transformers based AI to learn to play openage

# Setup
1.     Create enviroment and download requirements
 - `python -m venv env`
 - `source env/bin/activate && pip install -r requirements.txt` 

2.    Download deep-rts
 - `git submodule init`
 - `git submodule update`
 - `cd deep-rts && git checkout cace3c2c55ef3d86906ad492b0779d832df218c2`

3.    Compile deep-rts
 - `git submodule sync && git submodule update --init && pip install .`


# Extra commands

### To save all dependencies: `pip freeze > requirements.txt`
