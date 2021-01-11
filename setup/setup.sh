# create virtualenv and install Python modules
# change paths if you want

PATH_ENV=$HOME/env-mlim
PATH_REPO=$HOME/repos/mlim

python3 -m venv $PATH_ENV

source $PATH_ENV/bin/activate
pup install --upgrade pip
pip install -r $PATH_REPO/setup/requirements.txt
deactivate

source $PATH_ENV/bin/activate
jupyter serverextension enable --py jupyterlab_code_formatter
jupyter labextension install jupyterlab-plotly@4.14.1
deactivate
