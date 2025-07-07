# Use an official Python runtime as a parent image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.app.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.app.txt

# Install NLTK data
RUN python -m nltk.downloader stopwords

# Copy the current directory contents into the container at /app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
# CMD ["python", "app.py"]
