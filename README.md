Machine Learning Notebooks
==========================

This project aims at teaching you the fundamentals of Machine Learning in
python. It contains the example code and solutions to the exercises in the second edition of my O'Reilly book [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/):

<img src="https://images-na.ssl-images-amazon.com/images/I/51aqYc1QyrL._SX379_BO1,204,203,200_.jpg" title="book" width="150" />

**Note**: If you are looking for the first edition notebooks, check out [ageron/handson-ml](https://github.com/ageron/handson-ml).

## Quick Start

### Want to play with these notebooks online without having to install anything?
Use any of the following services.

**WARNING**: Please be aware that these services provide temporary environments: anything you do will be deleted after a while, so make sure you download any data you care about.

* **Recommended**: open this repository in [Colaboratory](https://colab.research.google.com/github/ageron/handson-ml2/blob/master/):
<a href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/"><img src="https://colab.research.google.com/img/colab_favicon.ico" width="90" /></a>

* Or open it in [Binder](https://mybinder.org/v2/gh/ageron/handson-ml2/master):
<a href="https://mybinder.org/v2/gh/ageron/handson-ml2/master"><img src="https://matthiasbussonnier.com/posts/img/binder_logo_128x128.png" width="90" /></a>

  * _Note_: Most of the time, Binder starts up quickly and works great, but when handson-ml2 is updated, Binder creates a new environment from scratch, and this can take quite some time.

* Or open it in [Deepnote](https://beta.deepnote.com/launch?template=data-science&url=https%3A//github.com/ageron/handson-ml2/blob/master/index.ipynb):
<a href="https://beta.deepnote.com/launch?template=data-science&url=https%3A//github.com/ageron/handson-ml2/blob/master/index.ipynb"><img src="https://www.deepnote.com/static/illustration.png" width="150" /></a>

### Just want to quickly look at some notebooks, without executing any code?

Browse this repository using [jupyter.org's notebook viewer](https://nbviewer.jupyter.org/github/ageron/handson-ml2/blob/master/index.ipynb):
<a href="https://nbviewer.jupyter.org/github/ageron/handson-ml2/blob/master/index.ipynb"><img src="https://jupyter.org/assets/nav_logo.svg" width="150" /></a>

_Note_: [github.com's notebook viewer](index.ipynb) also works but it is slower and the math equations are not always displayed correctly.

### Want to run this project using a Docker image?
Read the [Docker instructions](https://github.com/ageron/handson-ml2/tree/master/docker).

### Want to install this project on your own machine?

Start by installing [Anaconda](https://www.anaconda.com/distribution/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), [git](https://git-scm.com/downloads), and if you have a TensorFlow-compatible GPU, install the [GPU driver](https://www.nvidia.com/Download/index.aspx), as well as the appropriate version of CUDA and cuDNN (see TensorFlow's documentation for more details).

Next, clone this project by opening a terminal and typing the following commands (do not type the first `$` signs on each line, they just indicate that these are terminal commands):

    $ git clone https://github.com/ageron/handson-ml2.git
    $ cd handson-ml2

Next, run the following commands:

    $ conda env create -f environment.yml
    $ conda activate tf2
    $ python -m ipykernel install --user --name=python3

Finally, start Jupyter:

    $ jupyter notebook

If you need further instructions, read the [detailed installation instructions](INSTALL.md).

## Contributors
I would like to thank everyone who contributed to this project, either by providing useful feedback, filing issues or submitting Pull Requests. Special thanks go to Haesun Park and Ian Beauregard who reviewed every notebook and submitted many PRs, including help on some of the exercise solutions. Thanks as well to Steven Bunkley and Ziembla who created the `docker` directory, and to github user SuperYorio who helped on some exercise solutions.
