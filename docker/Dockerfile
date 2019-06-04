FROM continuumio/anaconda3:2019.03

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        libpq-dev \
        build-essential \
        git \
        sudo \
        cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev libboost-all-dev libsdl2-dev swig \
        unzip zip \
    && rm -rf /var/lib/apt/lists/*

RUN conda update -n base conda
RUN conda install -y \
        joblib \
        PyYAML==3.13
RUN conda install -y -c conda-forge \
        pyopengl \
        xgboost \
        nbdime
RUN pip install "urlextract"
RUN pip install "gym[atari,box2d,classic_control]"
RUN pip install "tensorflow-hub"
RUN pip install "tensorflow-serving-api"
RUN pip install "tfx"
#RUN pip install "tensorflow-addons"
RUN pip install "tf-agents-nightly"
RUN pip install "tfds-nightly"
RUN pip install "tfp-nightly"
RUN pip uninstall -y tensorflow
RUN pip uninstall -y tensorboard
RUN pip install "tf-nightly-2.0-preview"
RUN pip install "tb-nightly"


ARG username
ARG userid

ARG home=/home/${username}
ARG workdir=${home}/handson-ml2

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
    && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
    && chmod 0440 /etc/sudoers.d/${username}

WORKDIR ${workdir}
RUN chown ${username}:${username} ${workdir}

USER ${username}
WORKDIR ${workdir}

# The config below enables diffing notebooks with nbdiff (and nbdiff support
# in git diff command) after connecting to the container by "make exec" (or
# "docker-compose exec handson-ml2 bash")
#       You may also try running:
#         nbdiff NOTEBOOK_NAME.ipynb
#       to get nbdiff between checkpointed version and current version of the
# given notebook.

RUN git-nbdiffdriver config --enable --global

# INFO: Optionally uncomment any (one) of the following RUN commands below to ignore either
#       metadata or details in nbdiff within git diff
#RUN git config --global diff.jupyternotebook.command 'git-nbdiffdriver diff --ignore-metadata'
RUN git config --global diff.jupyternotebook.command 'git-nbdiffdriver diff --ignore-details'


COPY docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> ${home}/.bashrc
RUN echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc
RUN sudo rm /tmp/bashrc.bash


# INFO: Uncomment lines below to enable automatic save of python-only and html-only
#       exports alongside the notebook
#COPY docker/jupyter_notebook_config.py /tmp/
#RUN cat /tmp/jupyter_notebook_config.py >> ${home}/.jupyter/jupyter_notebook_config.py
#RUN sudo rm /tmp/jupyter_notebook_config.py


# INFO: Uncomment the RUN command below to disable git diff paging
#RUN git config --global core.pager ''


# INFO: Uncomment the RUN command below for easy and constant notebook URL (just localhost:8888)
#       That will switch Jupyter to using empty password instead of a token.
#       To avoid making a security hole you SHOULD in fact not only uncomment but
#       regenerate the hash for your own non-empty password and replace the hash below.
#       You can compute a password hash in any notebook, just run the code:
#          from notebook.auth import passwd
#          passwd()
#       and take the hash from the output
#RUN mkdir -p ${home}/.jupyter && \
#    echo 'c.NotebookApp.password = u"sha1:c6bbcba2d04b:f969e403db876dcfbe26f47affe41909bd53392e"' \
#    >> ${home}/.jupyter/jupyter_notebook_config.py
