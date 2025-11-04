# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port 8080 (Cloud Run expects this)
ENV PORT=8080

# Start your Gradio app directly
CMD ["python", "app.py"]
