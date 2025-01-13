# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nltk
RUN pip install accelerate

# Copy the rest of the application code into the container
COPY . /app

ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"
ENV CUDA_VISIBLE_DEVICES="0"

# Make port 8004 available to the world outside this container
EXPOSE 8004

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]