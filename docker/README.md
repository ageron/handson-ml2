
# Hands-on Machine Learning in Docker :-)

This is the Docker configuration which allows you to run and tweak the book's notebooks without installing any dependencies on your machine!
OK, any except `docker`. With `docker-compose`. Well, you may also want `make` (but it is only used as thin layer to call a few simple `docker-compose` commands).

## Prerequisites

As stated, the two things you need is `docker` and `docker-compose`.

Follow the instructions on [Install Docker](https://docs.docker.com/engine/installation/) and [Install Docker Compose](https://docs.docker.com/compose/install/) for your environment if you haven't got `docker` already.

Some general knowledge about `docker` infrastructure might be useful (that's an interesting topic on its own) but is not strictly *required* to just run the notebooks.

## Usage

### Prepare the image (once)

Switch to `docker` directory here and run `make build` (or `docker-compose build`) to build your docker image. That may take some time but is only required once. Or perhaps a few times after you tweak something in a `Dockerfile`.

After the process is finished you have a `handson-ml` image, that will be the base for your experiments. You can confirm that looking on results of `docker images` command.

### Run the notebooks

Run `make run` (or just `docker-compose up`) to start the jupyter server inside the container (also named `handson-ml`, same as image). Just point your browser to <http://localhost:8888> or the URL printed on the screen and you're ready to play with the book's code!

The server runs in the directory containing the notebooks, and the changes you make from the browser will be persisted there.

You can close the server just by pressing `Ctrl-C` in terminal window.

### Run additional commands in container

Run `make exec` (or `docker-compose exec handson-ml bash`) while the server is running to run an additional `bash` shell inside the `handson-ml` container. Now you're inside the environment prepared within the image.

One of the usefull things that can be done there may be comparing versions of the notebooks using the `nbdiff` command if you haven't got `nbdime` installed locally (it is **way** better than plain `diff` for notebooks). See [Tools for diffing and merging of Jupyter notebooks]<https://github.com/jupyter/nbdime> for more details.

You may also try `nbd NOTEBOOK_NAME.ipynb` command (custom, defined in the Dockerfile) to compare one of your notebooks with its `checkpointed` version. To be precise, the output will tell you "what modifications should be re-played on the *manually saved* version of the notebook (located in `.ipynb_checkpoints` subdirectory) to update it to the *current* i.e. *auto-saved* version (given as command's argument - located in working directory)".
