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
train: 
	python train.py --train_dir="C:\Users\mathe\Documents\random_split"

# Run the evaluate command inside the container
evaluate: 
	python evaluate.py --train_dir="C:\Users\mathe\Documents\random_split" --model_weights_file_path="D:\instadeep\saved_models\test.pth"

# Run the predict command inside the container
predict: 
	python predict.py --train_dir="C:\Users\mathe\Documents\random_split" --model_weights_file_path="D:\instadeep\saved_models\test.pth"
# Test functions command inside the container
test:
	python -m unittest discover -s tests/ -p "test*.py"