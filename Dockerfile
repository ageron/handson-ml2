FROM andrewosh/binder-base

USER root

RUN apt-get update -y &&\
    apt-get install --fix-missing -y \
        python-numpy\
        python-dev\
        cmake\
        zlib1g-dev\
        libjpeg-dev\
        xvfb\
        libav-tools\
        xorg-dev\
        python-opengl\
        libboost-all-dev\
        libsdl2-dev\
        swig &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*tmp

USER main

# Python 2
RUN conda install -c jjhelmus tensorflow=0.10.0
RUN conda install -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable toc2/main
RUN pip install --upgrade gym
RUN pip install --upgrade 'gym[atari]'

# Python 3
RUN conda install -n python3 -c jjhelmus tensorflow=0.10.0
RUN /bin/bash -c "source activate python3 && pip install --upgrade gym"
RUN /bin/bash -c "source activate python3 && pip install --upgrade 'gym[atari]'"
