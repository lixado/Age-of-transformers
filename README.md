# Age-of-transformers
Transformers based AI to learn to play deep-rts

# Index
<!-- vscode-markdown-toc -->
* [Setup](#Setup)
	* [Ubuntu](#Ubuntu)
		* [Create enviroment and download dependencies](#Createenviromentanddownloaddependencies)
		* [Download and compile deep-rts](#Downloadandcompiledeep-rts)
	* [Windows](#Windows)
	* [Mac](#Mac)
	*  [Docker](#Docker)
*  [TODOs](#TODOs)
*  [Extra commands](#Extracommands)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  <a name='Setup'></a>Setup

###  <a name='Ubuntu'></a>Ubuntu
####  <a name='Createenviromentanddownloaddependencies'></a>Create enviroment and download dependencies
```
python -m venv env
source env/bin/activate && pip install -r requirements.txt 
sudo xargs apt-get install -y <packages.txt
```

####  <a name='Downloadandcompiledeep-rts'></a>Download and compile deep-rts
```
git submodule init && git submodule update
cd deep-rts && git submodule init && git submodule update
pip install .
```


###  <a name='Windows'></a>Windows
TODO

###  <a name='Mac'></a>Mac
TODO

###  <a name='Docker'></a>Docker
* Pull all submodules before building
```
sudo docker build .
sudo docker run --gpus all -it --entrypoint bash <image hash> # runs with cuda
sudo docker container cp <container_hash>:/results/ ./results/ # copy files to outside container
```

##  <a name='TODOs'></a>TODOs
TODO

##  <a name='Extracommands'></a>Extra commands

- To save all dependencies: `pip freeze > requirements.txt`
- Remove all pip packages: `pip freeze | xargs pip uninstall -y` if error `pip uninstall DeepRTS`
- To update submodules: `git submodule sync`
- Get files from server: `scp -r <username>@10.225.148.248:/home/pedron18/Age-of-transformers-pedro/results/ results/`