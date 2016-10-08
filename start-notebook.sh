#!/bin/bash
OPTS=""
echo "$HOME/notebooks/index.ipynb: " $HOME/notebooks/index.ipynb
if [ -e $HOME/notebooks/index.ipynb ]; then
  OPTS="$OPTS --NotebookApp.default_url=/tree/index.ipynb "
fi
if [ -e $HOME/.binder_start ]; then
  source $HOME/.binder_start
fi
CMD="$OPTS $@"
echo "CMD: " $CMD

# Run Jupyter with xvfb-run so that it can render the CartPole
# environment without crashing:
xvfb-run -s "-screen 0 1400x900x24" jupyter notebook $CMD

