FROM andrewosh/binder-base

USER root
RUN apt-get update

USER main

# Python 2
RUN conda install jupyter matplotlib numexpr numpy pandas Pillow protobuf psutil scikit-learn scipy sympy wheel
RUN conda install -c jjhelmus tensorflow=0.10.0
RUN conda install https://github.com/ipython-contrib/IPython-notebook-extensions/archive/master.zip

# Python 3
RUN conda install -n python3 jupyter matplotlib numexpr numpy pandas Pillow protobuf psutil scikit-learn scipy sympy wheel
RUN conda install -n python3 -c jjhelmus tensorflow=0.10.0
RUN conda install -n python 3 https://github.com/ipython-contrib/IPython-notebook-extensions/archive/master.zip
