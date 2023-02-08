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


##  1. <a name='Setup'></a>Setup

###  1.1. <a name='Ubuntu'></a>Ubuntu
####  1.1.1. <a name='Createenviromentanddownloaddependencies'></a>Create enviroment and download dependencies
```
python -m venv env
source env/bin/activate && pip install -r requirements.txt 
```

####  1.1.2. <a name='Downloadandcompiledeep-rts'></a>Download and compile deep-rts
```
git submodule init && git submodule update
cd deep-rts && git submodule init && git submodule update
sudo xargs apt-get install -y <packages.txt
pip install .
```


###  1.2. <a name='Windows'></a>Windows
TODO

###  1.3. <a name='Mac'></a>Mac
TODO

###  1.4. <a name='Docker'></a>Docker
sudo docker build .

##  2. <a name='TODOs'></a>TODOs

##  3. <a name='Extracommands'></a>Extra commands

- To save all dependencies: `pip freeze > requirements.txt`
- Remove all pip packages: `pip freeze | xargs pip uninstall -y` if error `pip uninstall DeepRTS`
- To update submodules: `git submodule sync`