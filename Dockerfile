FROM andrewosh/binder-base

USER root

RUN apt-get update -y &&\
    apt-get install --fix-missing -y \
        cmake\
        libjpeg-dev\
        libav-tools\
        libboost-all-dev\
        libsdl2-dev\
        python-dev\
        python-opengl\
        python-pip\
        python3-dev\
        python3-opengl\
        python3-pip\
        swig\
        xorg-dev\
        xvfb\
        zlib1g-dev &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*tmp

USER main

RUN /usr/bin/pip2 install --upgrade --user pip wheel
RUN /usr/bin/pip3 install --upgrade --user pip wheel

ENV PATH /home/main/.local/bin:$PATH

# Install scientific packages
RUN pip2 install --upgrade --user matplotlib numexpr numpy pandas Pillow protobuf psutil scipy scikit-learn sympy jupyter
RUN pip3 install --upgrade --user matplotlib numexpr numpy pandas Pillow protobuf psutil scipy scikit-learn sympy jupyter

# Install TensorFlow
RUN pip2 install --user --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
RUN pip3 install --user --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp34-cp34m-linux_x86_64.whl

# Install OpenAI gym
RUN pip2 install --user --upgrade gym
RUN pip3 install --user --upgrade gym
RUN pip2 install --user --upgrade 'gym[all]'
RUN pip3 install --user --upgrade 'gym[all]'

# Install Jupyter extensions
RUN pip3 install --user --upgrade https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master

RUN rm -rf /home/main/.cache

# Jupyter extensions
#RUN conda install -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable toc2/main

RUN /home/main/anaconda2/bin/jupyter kernelspec remove -f python3
RUN /usr/bin/python2 -m ipykernel install --user
RUN /usr/bin/python3 -m ipykernel install --user

ADD .binder_start /home/main/
