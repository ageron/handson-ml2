FROM continuumio/anaconda3

WORKDIR /usr/src/project
COPY . /usr/src/project

RUN apt-get update && apt-get upgrade -y \
	
	&& apt-get install -y \
	 	libpq-dev \
	 	build-essential \
	 	git \

	&& rm -rf /var/lib/apt/lists/* \

	&& conda install -y -c conda-forge tensorflow=1.0.0 \
	&& conda install -y -c conda-forge jupyter_contrib_nbextensions \

	&& jupyter contrib nbextension install --user \
	&& jupyter nbextension enable toc2/main 
