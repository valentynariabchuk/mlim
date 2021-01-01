# Pipeline

# activate virtualenv and change path to repo
source $PATH_ENV/bin/activate
cd $PATH_REPO

# prepare data
python -m homework.h02.data

# baseline solution
python -m homework.h02.baseline-random
python -m homework.h02.baseline-heuristic

# train model
python -m homework.h02.model

# evaluation
python -m homework.h02.benchmark
