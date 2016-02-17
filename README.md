Machine Learning Notebooks
==========================

This project aims at teaching you the fundamentals of Machine Learning in
python. 

Simply open the [Jupyter](http://jupyter.org/) notebooks you are interested in:

* using [Binder](http://mybinder.org/): [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/ageron/ml-notebooks)
    * this let's you experiment with the code examples
* or directly within github (start at [index.ipynb](https://github.com/ageron/ml-notebooks/blob/master/index.ipynb))
* or by cloning this repository and running Jupyter locally.
    * if you prefer this option, follow the installation instructions below.

# Installation

You will need [git](https://git-scm.com/), obviously.

First, clone this repository:

    $ cd {*your development directory*}
    $ git clone https://github.com/ageron/ml-notebooks.git
    $ cd ml-notebooks

If you want an isolated environment, you can use [virtualenv](https://virtualenv.readthedocs.org/en/latest/):

    $ virtualenv env
    $ source ./env/bin/activate

Next, install the required python packages using pip:

    $ pip install -r requirements.txt         # On Linux
    OR
    $ pip install -r requirements_macosx.txt  # On MacOSX

Finally, launch Jupyter:

    $ jupyter notebook

This should start the Jupyter server locally, and open a browser window. Click on `index.ipynb` to get started.

