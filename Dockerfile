FROM python:3.9.13

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Upgrade pip
RUN pip install --upgrade pip

# Install MLflow and Cloudpickle
RUN pip install mlflow==2.21.2 cloudpickle==3.1.1 

# Install missing dependencies
RUN pip install --no-cache-dir attrs==25.1.0 defusedxml==0.7.1 ipython==8.18.1 matplotlib==3.9.4 more-itertools==10.6.0 \
    numpy==2.0.2 opencv-python==4.11.0.86 pandas==2.2.3 psutil==7.0.0 pydot==3.0.4 \
    scikit-learn==1.6.1 scipy==1.13.1 tensorflow-intel

# Set working directory
WORKDIR /app

# Copy source files
COPY . /app

# Copy models
COPY ./models /app/models

# Expose MLflow port
EXPOSE 5001

# Start MLflow model serving
CMD ["mlflow", "models", "serve", "--host", "0.0.0.0", "--port", "5001", "--model-uri", "file:///app/models/regression", "--no-conda"]

