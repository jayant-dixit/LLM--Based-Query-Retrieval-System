# Use official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Add this line before CMD
ENV PINECONE_API_KEY=pcsk_6WZck3_FJ6yG2hzRZBqHnXMdwbYFGBmvAQcGSifX58GuGwynBC1yLyBWUiWneCVjbChcJA


# Run the app
CMD ["python", "main.py"]
