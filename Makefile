# Copied from https://github.com/shreyashankar/create-ml-app

.PHONY: help lint run
# Makefile variables
VENV_NAME:=venv
PYTHON=${VENV_NAME}/bin/python3

# Include your variables here
RANDOM_SEED:=42
NUM_EPOCHS:=15
INPUT_DIM:=784
HIDDEN_DIM:=128
OUTPUT_DIM:=10

.DEFAULT: help
help:
	@echo "make venv"
	@echo "       prepare development environment, use only once"
	@echo "make lint"
	@echo "       run pylint"
	@echo "make run"
	@echo "       run project"

# Build the Docker image
build:
	docker build -t my_image .

lint: venv
	python -m pylint main.py

# Run the train command inside the container
train: build
	docker run --rm -v $(PWD):/app -it my_image python train.py --param1 value1 --param2 value2

# Run the evaluate command inside the container
evaluate: build
	docker run --rm -v $(PWD):/app -it my_image python evaluate.py --param1 value1 --param2 value2

# Run the predict command inside the container
predict: build
	docker run --rm -v $(PWD):/app -it my_image python predict.py --param1 value1 --param2 value2

test:
	python -m unittest discover -s tests/ -p "test*.py"