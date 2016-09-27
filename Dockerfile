FROM andrewosh/binder-base

USER root
RUN apt-get update

USER main

# Python 2
RUN conda install -c jjhelmus tensorflow=0.10.0
RUN conda install -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user

# Python 3
RUN conda install -n python3 -c jjhelmus tensorflow=0.10.0
