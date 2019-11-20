# Installation

## Download this repository
To install this repository and run the Jupyter notebooks on your machine, you will first need git, which you probably already have. Open a terminal and type `git` to check. If you do not have git, you can download it from [git-scm.com](https://git-scm.com/).

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ageron/handson-ml2.git
    $ cd handson-ml2

If you do not want to install git, you can instead download [master.zip](https://github.com/ageron/handson-ml2/archive/master.zip), unzip it, rename the resulting directory to `handson-ml2` and move it to your development directory.

## Python 3 and the required libraries
Next, you will need Python 3.6+ and a bunch of Python libraries. The simplest way to install these is to use Anaconda, which is a great cross-platform Python distribution for scientific computing. It comes bundled with many scientific libraries, including NumPy, Pandas, Matplotlib, Scikit-Learn and much more, so it's quite a large installation. If you choose to [download and install Anaconda](https://www.anaconda.com/distribution/), just make sure to install the Python 3 version. If you prefer a lighter weight Anaconda distribution, you can [install Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains the bare minimum to run the `conda` packaging tool.

Once Anaconda or Miniconda is installed, open a terminal and run the following command. It will create a new `conda` enviromnent containing every library you will need (by default, the environment will be named `tf2`, but you can choose another name using the `-n` option):

    $ conda env create -f environment.yml

Next, activate the new environment:

    $ conda activate tf2

> **Note**: if you don't like Anaconda for some reason, then you can install Python 3 and the required libraries manually (this is not recommended, unless you know what you are doing). For this, go through the [Manual Python Installation](#manual-python-installation) section then come back here and continue to the next sections.

## Using a GPU
If you have a TensorFlow-compatible GPU card (NVidia card with Compute Capability ≥ 3.5), and you want TensorFlow to use it, then you should follow TensorFlow's [GPU installation instructions](https://tensorflow.org/install/gpu) to install the driver and libraries such as CUDA and CuDNN.

Also make sure to replace the `tensorflow` library with the `tensorflow-gpu` library (this will no longer be needed starting in TensorFlow 2.1):

    $ pip uninstall tensorflow
    $ pip install -U tensorflow-gpu

## Reinforcement Learning Chapter Requirements
If you want to go through chapter 18 on Reinforcement Learning, you will need to [install OpenAI gym](https://gym.openai.com/docs) and its dependencies for Atari simulations.

## Starting Jupyter
You're almost there! You just need to register the `tf2` conda environment to Jupyter. The notebooks in this project will defaut to the environment named `python3`, so it's best to register this environment using the name `python3` (if you prefer to use another name, you will have to select it in the "Kernel > Change kernel..." menu in Jupyter every time you open a notebook):

    $ conda activate tf2
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

## Manual Python Installation
**Not recommended**: use Anaconda or Miniconda instead, unless you know what you're doing.

First, you will need Python 3.6 or 3.7. Some systems have it preinstalled. You can check by running the following command:

    $ python3 --version

If you have Python 3.6 or 3.7 already installed, that's great. If not, on Windows or MacOSX, you can just download it from [python.org](https://www.python.org/downloads/). On MacOSX, you can alternatively use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). If you are using Python 3.6+ on MacOSX, you need to run the following command to install the `certifi` package of certificates because Python 3.6+ on MacOSX has no certificates to validate SSL connections (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command # or Python 3.7

On Linux, unless you know what you are doing, you should probably use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3 python3-pip

Next, you will need to install the Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter and TensorFlow (and a few others). For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages.

In this section, we will look at how to use pip. First, make sure you have the latest version of pip installed:

    $ python3 -m pip install --user -U pip setuptools

The `--user` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use `sudo python3 -m pip` instead of `python3 -m pip` on Linux), and you should remove the `--user` option. The same is true of the command below that uses the `--user` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g., one for this project), with potentially very different libraries, and different versions:

    $ python3 -m pip install --user -U virtualenv
    $ python3 -m virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace `` `which python3` `` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

    $ source ./env/bin/activate

On Windows, the command is slightly different:

    $ .\env\Scripts\activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using `sudo python3` instead of `python3` on Linux).

    $ python3 -m pip install -U -r requirements.txt

Great! You now have Python 3.6 or 3.7 installed with all the required libraries. You can now resume the installation instructions starting at the [Using a GPU](#using-a-gpu) section above. However, you will need to replace the `pip` command with `python3 -m pip`, and replace `conda activate tf2` with `source ./env/bin/activate` (or `.\env\Scripts\activate` on Windows).