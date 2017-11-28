FROM continuumio/anaconda3

RUN apt-get update && apt-get upgrade -y \
	&& apt-get install -y \
		libpq-dev \
		build-essential \
		git \
		sudo \
	&& rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge \
		tensorflow=1.0.0 \
		jupyter_contrib_nbextensions

ARG username
ARG userid

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password && \
	echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} && \
	chmod 0440 /etc/sudoers.d/${username}

ENV HOME /home/${username}

WORKDIR ${HOME}/handson-ml
RUN chown ${username}:${username} ${HOME}/handson-ml

USER ${username}

RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable toc2/main


# INFO: Have RUN command below uncommented to for easy and constant URL (just localhost:8888)
#       (by setting empty password instead of using a token)
#       To avoid making a security hole the best would be to regenerate a hash for
#       your own non-empty password and to replace the hash below.
#       You can compute a password hash in the notebook, just run the code:
#          from notebook.auth import passwd
#          passwd()
RUN mkdir -p ${HOME}/.jupyter && \
	echo 'c.NotebookApp.password = u"sha1:c6bbcba2d04b:f969e403db876dcfbe26f47affe41909bd53392e"' \
	>> ${HOME}/.jupyter/jupyter_notebook_config.py


# INFO: Below - work in progress, nbdime not totally integrated, still it enables diffing
#       notebooks via nbdiff after connecting to container by "make exec" (docker exec)
#  Use:
#      nbd NOTEBOOK_NAME.ipynb
#    to get nbdiff between checkpointed version and current version of the given notebook
USER root
WORKDIR /
RUN conda install -y -c conda-forge nbdime
USER ${username}
WORKDIR ${HOME}/handson-ml


COPY docker/bashrc /tmp/bashrc
RUN cat /tmp/bashrc >> ${HOME}/.bashrc
RUN sudo rm -rf /tmp/bashrc
