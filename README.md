Machine Learning Notebooks
==========================

[![Gitter](https://badges.gitter.im/ageron/handson-ml.svg)](https://gitter.im/ageron/handson-ml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/ageron/handson-ml)

This project aims at teaching you the fundamentals of Machine Learning in
python. It contains the example code and solutions to the exercises in my O'Reilly book [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do):

[![book](http://akamaicovers.oreilly.com/images/0636920052289/rc_cat.gif)](http://shop.oreilly.com/product/0636920052289.do)

Simply open the [Jupyter](http://jupyter.org/) notebooks you are interested in:

* using Binder (recommended): [launch binder](http://mybinder.org/repo/ageron/handson-ml)
    * no installation needed, you can immediately experiment with the code examples
* or using [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb)
    * note: [github.com's notebook viewer](https://github.com/ageron/handson-ml/blob/master/index.ipynb) also works but it is slower and the math formulas are not displayed correctly
* or by cloning this repository and running Jupyter locally
    * if you prefer this option, follow the installation instructions below.

# Installation

No installation is required, just click the *launch binder* button above, this creates a new VM with everything you need already preinstalled, so you'll be good to go in a just a few seconds! But if you insist, here's how to install these notebooks on your own system.

Obviously, you will need [git](https://git-scm.com/) and [python](https://www.python.org/downloads/) (python 3 is recommended, but python 2 should work as well).

First, clone this repository:

    $ cd {your development directory}
    $ git clone https://github.com/ageron/handson-ml.git
    $ cd handson-ml

If you want an isolated environment (recommended), you can use [virtualenv](https://virtualenv.readthedocs.org/en/latest/):

    $ virtualenv env
    $ source ./env/bin/activate

There are different packages for TensorFlow, depending on your platform. Please edit `requirements.txt` and make sure only the right one for your platform is uncommented. Default is Python 3.5, Ubuntu/Linux 64-bits, CPU-only.

Also, if you want to go through chapter 16 on Reinforcement Learning, you will need to [install OpenAI gym](https://gym.openai.com/docs) and its dependencies for Atari simulations.

Then make sure pip is up to date, and use it to install the required python packages:

    $ pip install --upgrade pip
    $ pip install --upgrade -r requirements.txt

If you prefer to use [Anaconda](https://www.continuum.io/), you can run the following commands instead:

    $ conda install -c jjhelmus tensorflow=0.10.0
    $ conda install -c conda-forge jupyter_contrib_nbextensions

If you want to install the Jupyter extensions, run the following command:

    $ jupyter contrib nbextension install --user

Then you can activate an extension, such as the Table of Contents (2) extension:

    $ jupyter nbextension enable toc2/main

Finally, launch Jupyter:

    $ jupyter notebook

This should start the Jupyter server locally, and open your browser. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started. You can visit [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions) to activate and configure Jupyter extensions.

That's it! Have fun learning ML.
