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
You can then run these commands by running make train, make test, make evaluate, and make predict on your local machine. The build target is a dependency for all other targets, so it will be run automatically if necessary. The --param1 value1 --param2 value2 are examples of parameters that can be passed to the command.

Makefile is a simple way to automate repetitive tasks, such as building and deploying code, while the Dockerfile is used to build an image of your application that can run in a container. A makefile allows you to run specific commands (like building, testing and deploying) in the context of the current project, while Dockerfiles are used to create a consistent and reproducible environment for your application to run in.

test:
	python -m unittest discover -s tests/ -p "test*.py"