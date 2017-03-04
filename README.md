Machine Learning Notebooks
==========================

This project aims at teaching you the fundamentals of Machine Learning in
python. It contains the example code and solutions to the exercises in my O'Reilly book [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do):

[![book](http://akamaicovers.oreilly.com/images/0636920052289/rc_cat.gif)](http://shop.oreilly.com/product/0636920052289.do)

Simply open the [Jupyter](http://jupyter.org/) notebooks you are interested in:

* Using [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb)
    * note: [github.com's notebook viewer](https://github.com/ageron/handson-ml/blob/master/index.ipynb) also works but it is slower and the math formulas are not displayed correctly
* or by cloning this repository and running Jupyter locally
    * if you prefer this option, follow the installation instructions below.

# Installation

Obviously, you will need [git](https://git-scm.com/) and [python](https://www.python.org/downloads/) (python 3 is recommended, but python 2 should work as well).

First, clone this repository:

    $ cd {your development directory}
    $ git clone https://github.com/ageron/handson-ml.git
    $ cd handson-ml

If you want an isolated environment (recommended), you can use [virtualenv](https://virtualenv.readthedocs.org/en/latest/):

    $ virtualenv env
    $ source ./env/bin/activate

If you want to go through chapter 16 on Reinforcement Learning, you will need to [install OpenAI gym](https://gym.openai.com/docs) and its dependencies for Atari simulations.

Then make sure pip is up to date, and use it to install the required python packages:

    $ pip install --upgrade pip
    $ pip install --upgrade -r requirements.txt

If you prefer to use [Anaconda](https://www.continuum.io/), you can run the following commands instead:

    $ conda install -c conda-forge tensorflow=1.0.0
    $ conda install -c conda-forge jupyter_contrib_nbextensions

If you want to install the Jupyter extensions, run the following command:

    $ jupyter contrib nbextension install --user

Then you can activate an extension, such as the Table of Contents (2) extension:

    $ jupyter nbextension enable toc2/main

Finally, launch Jupyter:

    $ jupyter notebook

This should start the Jupyter server locally, and open your browser. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started. You can visit [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions) to activate and configure Jupyter extensions.

That's it! Have fun learning ML.
