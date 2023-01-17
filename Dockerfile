# Use a base image with Python and the required dependencies installed
FROM python:3.10-slim-buster

# Copy the requirements.txt file
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt
