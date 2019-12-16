# Installation

## Download this repository
To install this repository and run the Jupyter notebooks on your machine, you will first need git, which you may already have. Open a terminal and type `git` to check. If you do not have git, you can download it from [git-scm.com](https://git-scm.com/).

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ageron/handson-ml2.git
    $ cd handson-ml2

If you do not want to install git, you can instead download [master.zip](https://github.com/ageron/handson-ml2/archive/master.zip), unzip it, rename the resulting directory to `handson-ml2` and move it to your development directory.

## Install Anaconda
Next, you will need Python 3.6 or 3.7 and a bunch of Python libraries. The simplest way to install these is to use Anaconda, which is a great cross-platform Python distribution for scientific computing. It comes bundled with many scientific libraries, including NumPy, Pandas, Matplotlib, Scikit-Learn and much more, so it's quite a large installation. If you choose to [download and install Anaconda](https://www.anaconda.com/distribution/), just make sure to install the Python 3 version. If you prefer a lighter weight Anaconda distribution, you can [install Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains the bare minimum to run the `conda` packaging tool.

Once Anaconda or miniconda is installed, then run the following command to update the conda packaging tool to the latest version:

    $ conda update -n base -c defaults conda

> **Note**: if you don't like Anaconda for some reason, then you can install Python 3 and use pip to install the required libraries manually (this is not recommended, unless you know what you are doing).


## Install the GPU Driver and Libraries
If you have a TensorFlow-compatible GPU card (NVidia card with Compute Capability ≥ 3.5), and you want TensorFlow to use it, then you should download the latest driver for your card from [nvidia.com](https://www.nvidia.com/Download/index.aspx?lang=en-us) and install it. You will also need NVidia's CUDA and cuDNN libraries, but the good news is that they will be installed automatically when you install the tensorflow-gpu package from Anaconda. However, if you don't use Anaconda, you will have to install them manually. If you hit any roadblock, see TensorFlow's [GPU installation instructions](https://tensorflow.org/install/gpu) for more details.

If you want to use a GPU then you should also edit environment.yml (or environment-windows.yml if you're on Windows), located at the root of the handson-ml2 project, replace tensorflow=2.0.0 with tensorflow-gpu=2.0.0, and replace tensorflow-serving-api==2.0.0 with tensorflow-serving-api-gpu==2.0.0. This will not be needed anymore when TensorFlow 2.1 is released.

## Create the tf2 Environment
Next, make sure you're in the handson-ml2 directory and run the following command. It will create a new `conda` enviromnent containing every library you will need to run all the notebooks (by default, the environment will be named `tf2`, but you can choose another name using the `-n` option):

    $ conda env create -f environment.yml # or environment-windows.yml on Windows

Next, activate the new environment:

    $ conda activate tf2

## Windows
If you're on Windows, and you want to go through chapter 18 on Reinforcement Learning, then you will also need to run the following command. It installs a Windows-compatible fork of the atari-py library.

    $ pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py


> **Warning**: TensorFlow Transform (used in chapter 13) and TensorFlow-AddOns (used in chapter 16) are not yet available on Windows, but the TensorFlow team is working on it.


## Start Jupyter
You're almost there! You just need to register the `tf2` conda environment to Jupyter. The notebooks in this project will defaut to the environment named `python3`, so it's best to register this environment using the name `python3` (if you prefer to use another name, you will have to select it in the "Kernel > Change kernel..." menu in Jupyter every time you open a notebook):

    $ python3 -m ipykernel install --user --name=python3

And that's it! You can now start Jupyter like this:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started.

Congrats! You are ready to learn Machine Learning, hands on!

When you're done with Jupyter, you can close it by typing Ctrl-C in the Terminal window where you started it. Every time you want to work on this project, you will need to open a Terminal, and run:

    $ cd $HOME # or whatever development directory you chose earlier
    $ cd handson-ml2
    $ conda activate tf2
    $ jupyter notebook
