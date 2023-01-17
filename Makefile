# Makefile variables
VENV_NAME:=venv
PYTHON=${VENV_NAME}/bin/python3

# Include your variables here
RANDOM_SEED:=42
NUM_EPOCHS:=15
INPUT_DIM:=784
HIDDEN_DIM:=128
OUTPUT_DIM:=10

# Build the Docker image
build:
	docker build -t my_image .

lint:
	python -m pylint instadeep-test

# Run the train command inside the container
train:
	python train.py --train_dir="$(train_dir)" --gpus=$(gpus)

# Run the minitrain command inside the container
minitrain:
	python train.py --train_dir="mini-dataset" --gpus=$(gpus)

# Run the evaluate command inside the container
evaluate:
	python evaluate.py --train_dir="$(train_dir)" --model_weights_file_path="$(model_weights_file_path)"

# Run the predict command inside the container
predict:
	python predict.py --train_dir="$(train_dir)" --model_weights_file_path="$(model_weights_file_path)"
# Test functions command inside the container
test:
	python -m unittest discover -s tests/ -p "test*.py"