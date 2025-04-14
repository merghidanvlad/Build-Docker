import mlflow
import mlflow.pyfunc
import os

# Set the tracking URI for MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Set the model URI and the destination directory
model_uri = "models:/regression_test_1/1"  # Replace with your model's name and version
model_dir = "./models/regression"  # Specify the directory where you want to save the model

# Download the model
model = mlflow.pyfunc.load_model(model_uri)

# Define a PythonModel subclass to wrap the model
class MyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Save the model to the specified directory
mlflow.pyfunc.save_model(path=model_dir, python_model=MyModel())
