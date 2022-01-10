# Installation

## Download this repository
To install this repository and run the Jupyter notebooks on your machine, you will first need git, which you may already have. Open a terminal and type `git` to check. If you do not have git, you can download it from [git-scm.com](https://git-scm.com/).

Next, clone this repository by opening a terminal and typing the following commands (do not type the first `$` on each line, it's just a convention to show that this is a terminal prompt, not something else like Python code):

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ageron/handson-ml2.git
    $ cd handson-ml2

If you do not want to install git, you can instead download [master.zip](https://github.com/ageron/handson-ml2/archive/master.zip), unzip it, rename the resulting directory to `handson-ml2` and move it to your development directory.

## Install Anaconda
Next, you will need Python 3 and a bunch of Python libraries. The simplest way to install these is to [download and install Anaconda](https://www.anaconda.com/distribution/), which is a great cross-platform Python distribution for scientific computing. It comes bundled with many scientific libraries, including NumPy, Pandas, Matplotlib, Scikit-Learn and much more, so it's quite a large installation. If you prefer a lighter weight Anaconda distribution, you can [install Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains the bare minimum to run the `conda` packaging tool. You should install the latest version of Anaconda (or Miniconda) available.

During the installation on MacOSX and Linux, you will be asked whether to initialize Anaconda by running `conda init`: you should accept, as it will update your shell script to ensure that `conda` is available whenever you open a terminal. After the installation, you must close your terminal and open a new one for the changes to take effect.

During the installation on Windows, you will be asked whether you want the installer to update the `PATH` environment variable. This is not recommended as it may interfere with other software. Instead, after the installation you should open the Start Menu and launch an Anaconda Shell whenever you want to use Anaconda.

Once Anaconda (or Miniconda) is installed, run the following command to update the `conda` packaging tool to the latest version:

    $ conda update -n base -c defaults conda

> **Note**: if you don't like Anaconda for some reason, then you can install Python 3 and use pip to install the required libraries manually (this is not recommended, unless you really know what you are doing). I recommend using Python 3.8, since some libs don't support Python 3.9 or 3.10 yet.


## Install the GPU Driver and Libraries
If you have a TensorFlow-compatible GPU card (NVidia card with Compute Capability ≥ 3.5), and you want TensorFlow to use it, then you should download the latest driver for your card from [nvidia.com](https://www.nvidia.com/Download/index.aspx?lang=en-us) and install it. You will also need NVidia's CUDA and cuDNN libraries, but the good news is that they will be installed automatically when you install the tensorflow-gpu package from Anaconda. However, if you don't use Anaconda, you will have to install them manually. If you hit any roadblock, see TensorFlow's [GPU installation instructions](https://tensorflow.org/install/gpu) for more details.

## Create the `tf2` Environment
Next, make sure you're in the `handson-ml2` directory and run the following command. It will create a new `conda` environment containing every library you will need to run all the notebooks (by default, the environment will be named `tf2`, but you can choose another name using the `-n` option):

    $ conda env create -f environment.yml

Next, activate the new environment:

    $ conda activate tf2


## Start Jupyter
You're almost there! You just need to register the `tf2` conda environment to Jupyter. The notebooks in this project will default to the environment named `python3`, so it's best to register this environment using the name `python3` (if you prefer to use another name, you will have to select it in the "Kernel > Change kernel..." menu in Jupyter every time you open a notebook):

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

## Update This Project and its Libraries
I regularly update the notebooks to fix issues and add support for new libraries. So make sure you update this project regularly.

For this, open a terminal, and run:

    $ cd $HOME # or whatever development directory you chose earlier
    $ cd handson-ml2 # go to this project's directory
    $ git pull

If you get an error, it's probably because you modified a notebook. In this case, before running `git pull` you will first need to commit your changes. I recommend doing this in your own branch, or else you may get conflicts:

    $ git checkout -b my_branch # you can use another branch name if you want
    $ git add -u
    $ git commit -m "describe your changes here"
    $ git checkout master
    $ git pull

Next, let's update the libraries. First, let's update `conda` itself:

    $ conda update -c defaults -n base conda

Then we'll delete this project's `tf2` environment:

    $ conda activate base
    $ conda env remove -n tf2

And recreate the environment:

    $ conda env create -f environment.yml

Lastly, we reactivate the environment and start Jupyter:

    $ conda activate tf2
    $ jupyter notebook
