FROM andrewosh/binder-base

USER root
RUN apt-get update
RUN apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

USER main

# Python 2
RUN conda install -c jjhelmus tensorflow=0.10.0
RUN conda install -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable toc2/main
RUN pip install gym gym[atari]

# Python 3
RUN conda install -n python3 -c jjhelmus tensorflow=0.10.0
RUN /bin/bash -c "source activate python3 && pip install gym gym[atari]"
