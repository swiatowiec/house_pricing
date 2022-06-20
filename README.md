# Docker container for training and calling a linear regression model based on the housing prices dataset to predict respective house prices.

## DATASET
data source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

## USAGE
```
usage: run_training_process.py [-h] [--artifacts_path ARTIFACTS_PATH] [--test-size TEST_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --artifacts_path ARTIFACTS_PATH
  --test-size TEST_SIZE
```

## RUN LOCALLY
1. Build docker image:
`docker build .`

2. Run training and save results in `/tmp/artifacts` folder on local machine
`docker run --mount type=bind,src=/tmp/artifacts,dst=/app/artifacts CONTAINER_ID`

3. Run tests:
`docker run 58 python tests.py`



