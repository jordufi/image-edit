FROM python:3.12-slim

WORKDIR /app

# Install nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY config.json .
COPY qwen-image.json .
COPY qwen-2images.json .
COPY qwen-3images.json .
COPY start.sh .

# Copy static files
COPY static/ /app/static/

# Copy nginx config
COPY nginx.conf /etc/nginx/sites-available/default

# Create outputs directory
RUN mkdir -p /app/outputs

# Make start script executable
RUN chmod +x start.sh

EXPOSE 9093

CMD ["./start.sh"]
