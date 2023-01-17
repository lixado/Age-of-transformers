# Age-of-transformers
Transformers based AI to learn to play openage

# Setup
1.     Create enviroment and download requirements
 - `python -m venv env`
 - `source env/bin/activate && pip install -r requirements.txt` 

2.    Download deep-rts
 - `git submodule init && git submodule update`setuptools-git-versioning==1.5.0
 - `cd deep-rts && git submodule sync && git submodule update --init && sudo pip install .`

3.    Compile deep-rts
 - `git submodule sync && git submodule update --init && pip install .`



1. Set setup.py
setuptools-git-versioning==1.5.0

2. install extra
apt-get install libdbus-1-dev libxi-dev libxtst-dev

3. change /home/lixado/.local/share/pmm/1.5.1/vcpkg-master/ports/xsimd/portfile.cmake
8.0.2 xsimd
415aeddd9818408ddd7dd62831e40667e1253deef9337422dc886d489babcf79e3d1e42beb6b024e77a48dc113109b45a6678bc7d55f5c2a3e359443a83270c1





Working fork
git clone git@github.com:diegogutierrez2001/deep-rts.git


# Extra commands

### To save all dependencies: `pip freeze > requirements.txt`
