import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import mlflow
import itertools

# Descărcăm setul de date Boston Housing
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print(df.shape)

# Definim variabilele independente (X) și variabila dependentă (y)
X = df.drop("medv", axis=1)
y = df["medv"]

# Împărțim setul de date în antrenare (80%) și testare (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardizăm datele pentru o performanță mai bună
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Regression_pret_casa_mlflow")

optimizers = ["sgd"]
batch_sizes = [16, 32]
epochs_list = [25]
neurons = [128, 256]
learning_rate = [0.001, 0.0001]

param_combinations = list(itertools.product(optimizers, batch_sizes, epochs_list, neurons, learning_rate))

for run_id, (optimizer, batch_size, epochs, neuron, learning_rate) in enumerate(param_combinations):

    # Build the model
    with mlflow.start_run():
        run_name = f"Run_{run_id+1}_Opt-{optimizer}_BS-{batch_size}_Ep-{epochs}_Neurons-{neuron}_LR-{learning_rate}"
        mlflow.set_tag("mlflow.runName", run_name)
        model = models.Sequential([ 
            layers.Dense(neuron, activation='relu', input_shape=(X_train.shape[1],)),  
            layers.Dense(neuron, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(
            # Optimizer used for training the model
            optimizer= optimizer,  # Adam is an optimization algorithm that combines the benefits of both Adagrad and RMSProp, providing faster convergence and efficient handling of sparse gradients.
            # Loss function used to measure how well the model's predictions match the true labels
            loss='mse',  
            metrics=['mae']  # mae is used as a metric to track the absolute value of our measuring unit - 1000 USD
        )
        optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("neurons", neuron)
        mlflow.log_param("learning rate", learning_rate)
    # Train the model and store the training history
        history = model.fit(
            # Input data for training
            X_train,  
            # Target data for training
            y_train,  
            # Number of epochs (iterations) to train the model
            epochs=epochs,  # The model will go through the training data x times (x epochs)
            # Data to evaluate the model on after each epoch (used to monitor performance during training)
            validation_data=(X_test, y_test),  # Uses the X_test and y_test datasets to evaluate model performance after each epoch
            # Number of samples per gradient update (batch size)
            batch_size = batch_size  # Defines how many samples will be processed before the model updates its weights
        )

        # Evaluate the model
        test_loss, test_mae = model.evaluate(X_test, y_test)
        # The 'evaluate' method tests the trained model on the test dataset.
        # It returns the loss value (how well the model performs) and accuracy (how many predictions were correct).

        print(f"Test mae: {test_mae}")

        # Make predictions on the test set
        # predictions = model.predict(X_test)
        # 'predict' method uses the trained model to make predictions on the test data.
        # It returns the model's predicted probabilities for each class.

        # Facem predicții pe setul de testare
        # y_pred = model.predict(X_test).flatten()
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_mae", test_mae)

        for epoch in range(epochs):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_mae", history.history['mae'][epoch], step=epoch)
            mlflow.log_metric("test_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("test_mae", history.history['val_mae'][epoch], step=epoch)

        mlflow.tensorflow.log_model(model, f"regression_model_run{run_id+1}")
        
        mlflow.end_run()