FROM andrewosh/binder-base

USER root

# Extra packages needs for OpenAI gym
RUN apt-get update -y &&\
    apt-get install --fix-missing -y \
        cmake\
        libav-tools\
        libboost-all-dev\
        libjpeg-dev\
        libsdl2-dev\
        python-dev\
        python-numpy\
        python-opengl\
        python-pip\
        python3-dev\
        python3-numpy\
        python3-opengl\
        python3-pip\
        swig\
        xorg-dev\
        xvfb\
        zlib1g-dev &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*tmp

USER main

# Add nbextensions to Jupyter and activate the Table of Contents
RUN conda install -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable toc2/main

# Use upgraded pip
RUN /usr/bin/pip install --upgrade --user pip wheel
RUN /usr/bin/pip3 install --upgrade --user pip wheel
ENV PATH /home/main/.local/bin:$PATH

# Install scientific packages
RUN pip install --upgrade --user matplotlib numexpr numpy pandas Pillow protobuf psutil scipy scikit-learn sympy
RUN pip3 install --upgrade --user matplotlib numexpr numpy pandas Pillow protobuf psutil scipy scikit-learn sympy

# Install TensorFlow
RUN pip install --upgrade --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
RUN pip3 install --upgrade --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl

# Install OpenAI gym
RUN pip install --upgrade --user gym
RUN pip3 install --upgrade --user gym
RUN pip install --upgrade --user "gym[atari]"
RUN pip3 install --upgrade --user "gym[atari]"

# Replace the default script to start Jupyter with xvfb-run so that OpenAI gym
# can render some of the environments (eg. CartPole-v0) without crashing when
# it tries to access the display on a headless server.
ADD start-notebook.sh /home/main/

RUN ipython2 kernel install --user
RUN ipython3 kernel install --user
