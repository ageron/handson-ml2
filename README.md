Machine Learning Notebooks
==========================

This project aims at teaching you the fundamentals of Machine Learning in
python. It contains the example code and solutions to the exercises in the second edition of my O'Reilly book [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/):

<img src="https://images-na.ssl-images-amazon.com/images/I/51tOhPQBmSL._SX379_BO1,204,203,200_.jpg" title="book" width="150" />

**Note**: If you are looking for the first edition notebooks, check out [ageron/handson-ml](https://github.com/ageron/handson-ml).

## Quick Start

### Want to play with these notebooks without having to install anything?
Use any of the following services.

**WARNING**: Please be aware that these services provide temporary environments: anything you do will be deleted after a while, so make sure you save anything you care about.

* Open this repository in [Colaboratory](https://colab.research.google.com/github/ageron/handson-ml2/blob/master/):
<a href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/"><img src="https://colab.research.google.com/img/colab_favicon.ico" width="90" /></a>

* Or open it in [Binder](https://mybinder.org/v2/gh/ageron/handson-ml2/master):
<a href="https://mybinder.org/v2/gh/ageron/handson-ml2/master"><img src="https://matthiasbussonnier.com/posts/img/binder_logo_128x128.png" width="90" /></a>

  * _Note_: Most of the time, Binder starts up quickly and works great, but when handson-ml2 is updated, Binder creates a new environment from scratch, and this can take quite some time.

* Or open it in [Deepnote](https://beta.deepnote.com/launch?template=data-science&url=https%3A//github.com/ageron/handson-ml2/blob/master/index.ipynb):
<a href="https://beta.deepnote.com/launch?template=data-science&url=https%3A//github.com/ageron/handson-ml2/blob/master/index.ipynb"><img src="https://www.deepnote.com/static/illustration.png" width="150" /></a>

### Just want to quickly look at some notebooks, without executing any code?

Browse this repository using [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/ageron/handson-ml2/blob/master/index.ipynb):
<a href="http://nbviewer.jupyter.org/github/ageron/handson-ml2/blob/master/index.ipynb"><img src="https://jupyter.org/assets/nav_logo.svg" width="150" /></a>

_Note_: [github.com's notebook viewer](https://github.com/ageron/handson-ml2/blob/master/index.ipynb) also works but it is slower and the math equations are not always displayed correctly.

### Want to install this project on your own machine?

If you have a working Python 3.5+ environment and git is installed, then this project and its dependencies can be installed with pip. Open a terminal and run the following commands (do not type the `$` signs, they just indicate that this is a terminal command):

    $ git clone https://github.com/ageron/handson-ml2.git
    $ cd handson-ml2
    $ python3 -m pip install --user --upgrade pip setuptools
    $ # Read `requirements.txt` if you want to use a GPU.
    $ python3 -m pip install --user --upgrade -r requirements.txt
    $ jupyter notebook

Or using Anaconda:

    $ git clone https://github.com/ageron/handson-ml2.git
    $ cd handson-ml2
    $ # Read `environment.yml` if you want to use a GPU.
    $ conda env create -f /tmp/environment.yml
    $ conda activate tf2
    $ jupyter notebook

If you need more detailed installation instructions, read the [detailed installation instructions](INSTALL.md).

## Contributors
I would like to thank everyone who contributed to this project, either by providing useful feedback, filing issues or submitting Pull Requests. Special thanks go to Haesun Park who helped on some of the exercise solutions, and to Steven Bunkley and Ziembla who created the `docker` directory.
