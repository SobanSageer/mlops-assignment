# Use official Python image
FROM python:3.10

# Set working directory in container
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command to run training
CMD ["python", "src/train.py"]
