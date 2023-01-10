# Age-of-transformers
Transformers based AI to learn to play openage




# Setup

## Ubuntu

1.     Install dependencies for openage
 - `sudo apt-get update`
 - `sudo apt-get install g++-10 cmake cython3 libeigen3-dev libepoxy-dev libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libogg-dev libopus-dev libopusfile-dev libpng-dev libsdl2-dev libsdl2-image-dev python3-dev python3-jinja2 python3-numpy python3-lz4 python3-pil python3-pip python3-pygments python3-toml qml-module-qtquick-controls qtdeclarative5-dev`

2.     Install dependencies for nyan
 - `sudo apt-get install cmake flex make`
 - `sudo apt-get install gcc g++`

3.     Clone repo
 - `git clone git@github.com:lixado/Age-of-transformers.git`
 - `cd Age-of-transformers`
 - `git submodule init`
 - `git submodule update`

4.     Compile openage
 - `cd openage && ./configure --mode=release --download-nyan --prefix=./../../../game && make && make install`

5.     Create enviroment and download requirements
 - `cd .. && python -m venv env`
 - `source env/bin/activate && pip install -r requirements.txt` 


# Extra commands

### To save all dependencies: `pip freeze > requirements.txt`
