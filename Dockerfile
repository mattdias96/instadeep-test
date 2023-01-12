# Use a base image with Python and the required dependencies installed
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script and helper script
COPY main.py helper.py .

# Copy the data and models directory
COPY data models .

# Start the training script and Tensorboard
CMD ["bash", "-c", "tensorboard --logdir=./logs --host 0.0.0.0 & python main.py"]
