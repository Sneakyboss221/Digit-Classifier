# 🚀 Deployment Guide

This guide covers deploying the Ultimate MNIST Digit Recognition project to various platforms and environments.

## 📋 Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Logging](#monitoring-and-logging)

## 🏠 Local Development

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/Digit-Classifier.git
cd Digit-Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Quick setup
python quick_start.py

# Launch application
streamlit run interface/streamlit_app.py
```

### Using Makefile
```bash
# Install development dependencies
make install-dev

# Setup complete environment
make setup

# Run web application
make run-web

# Run tests
make test
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p saved_models results

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  mnist-classifier:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./saved_models:/app/saved_models
      - ./results:/app/results
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run
```bash
# Build image
docker build -t mnist-classifier .

# Run container
docker run -p 8501:8501 mnist-classifier

# Using Docker Compose
docker-compose up -d
```

## ☁️ Cloud Deployment

### Heroku

#### Procfile
```
web: streamlit run interface/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

#### runtime.txt
```
python-3.10.6
```

#### Deployment Steps
```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-mnist-app

# Add buildpacks
heroku buildpacks:add heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Google Cloud Platform (GCP)

#### app.yaml
```yaml
runtime: python310
entrypoint: streamlit run interface/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0

env_variables:
  PYTHONPATH: "/app"

automatic_scaling:
  target_cpu_utilization: 0.6
  min_instances: 1
  max_instances: 10

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10
```

#### Deployment Steps
```bash
# Install Google Cloud SDK
# Initialize project
gcloud init

# Deploy to App Engine
gcloud app deploy
```

### AWS

#### Elastic Beanstalk
```yaml
# .ebextensions/01_packages.config
packages:
  yum:
    gcc: []
    gcc-c++: []

# .ebextensions/02_requirements.config
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: interface/streamlit_app.py
```

#### Deployment Steps
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init

# Create environment
eb create mnist-classifier-env

# Deploy
eb deploy
```

### Azure

#### Azure App Service
```yaml
# .deployment
[config]
command = pip install -r requirements.txt && streamlit run interface/streamlit_app.py --server.port=8000 --server.address=0.0.0.0
```

#### Deployment Steps
```bash
# Install Azure CLI
# Login to Azure
az login

# Create resource group
az group create --name mnist-rg --location eastus

# Create app service plan
az appservice plan create --name mnist-plan --resource-group mnist-rg --sku B1

# Create web app
az webapp create --name mnist-classifier --resource-group mnist-rg --plan mnist-plan --runtime "PYTHON|3.10"

# Deploy
az webapp deployment source config-local-git --name mnist-classifier --resource-group mnist-rg
```

## 🏭 Production Considerations

### Environment Variables
```bash
# Production settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Model settings
export MODEL_CACHE_DIR=/app/saved_models
export RESULTS_DIR=/app/results
export LOG_LEVEL=INFO
```

### Security
- Use HTTPS in production
- Implement authentication if needed
- Set up proper firewall rules
- Use environment variables for sensitive data
- Regular security updates

### Performance
- Use GPU instances for training
- Implement caching for model predictions
- Optimize image preprocessing
- Monitor resource usage

### Scaling
- Use load balancers for multiple instances
- Implement horizontal scaling
- Use CDN for static assets
- Database scaling if needed

## 📊 Monitoring and Logging

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks
```python
# Add to streamlit app
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

### Metrics
- Request latency
- Model prediction accuracy
- Resource utilization
- Error rates
- User engagement

### Monitoring Tools
- **Application**: New Relic, DataDog, AppDynamics
- **Infrastructure**: CloudWatch, Azure Monitor, Stackdriver
- **Logs**: ELK Stack, Splunk, Papertrail

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port
   lsof -i :8501
   # Kill process
   kill -9 <PID>
   ```

2. **Memory issues**
   ```bash
   # Increase memory limit
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   ```

3. **Model loading errors**
   ```bash
   # Check model files
   ls -la saved_models/
   # Re-download models
   python download_pretrained_models.py
   ```

### Debug Mode
```bash
# Enable debug logging
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run interface/streamlit_app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
- Check the [GitHub Issues](https://github.com/yourusername/Digit-Classifier/issues)
- Review the [Documentation](https://github.com/yourusername/Digit-Classifier#readme)
- Contact the maintainers

---

**Happy Deploying! 🚀**
