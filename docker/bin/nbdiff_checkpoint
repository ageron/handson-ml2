#!/bin/bash
if [[ "$#" -lt 1 || "$1" =~ ^((-h)|(--help))$ ]] ; then
    echo "usage: nbdiff_checkpoint NOTEBOOK.ipynb"
    echo
    echo "Show differences between given jupyter notebook and its checkpointed version (in .ipynb_checkpoints subdirectory)"
    exit
fi

DIRNAME=$(dirname "$1")
BASENAME=$(basename "$1" .ipynb)
shift

WORKING_COPY=$DIRNAME/$BASENAME.ipynb
CHECKPOINT_COPY=$DIRNAME/.ipynb_checkpoints/$BASENAME-checkpoint.ipynb

echo "----- Analysing how to change $CHECKPOINT_COPY into $WORKING_COPY -----"
nbdiff "$CHECKPOINT_COPY" "$WORKING_COPY" --ignore-details "$@"
