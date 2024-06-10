import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation='sigmoid', output_activation='sigmoid', learning_rate=0.01, seed=None, dropout_rate=0.0, l2_lambda=0.0, batch_size=1):
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def activate(self, x, activation):
        if activation == 'sigmoid':
            return self.sigmoid(x)
        elif activation == 'relu':
            return self.relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def activate_derivative(self, x, activation):
        if activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif activation == 'relu':
            return self.relu_derivative(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, inputs, training=True):
        self.activations = [inputs]
        self.z_values = []

        for i in range(self.num_layers - 2):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.activate(z, self.hidden_activation)
            if training and self.dropout_rate > 0:
                activation = self.dropout(activation, self.dropout_rate)
            self.activations.append(activation)

        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.activate(z, self.output_activation)
        self.activations.append(output)

        return self.activations[-1]

    def dropout(self, x, rate):
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask

    def backward(self, inputs, targets):
        # Compute the output error and delta
        output_error = targets - self.activations[-1]
        output_delta = output_error * self.activate_derivative(self.activations[-1], self.output_activation)
        
        # Initialize deltas and errors
        deltas = [output_delta]
        
        # Compute deltas for hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(deltas[-1], self.weights[i].T)
            delta = error * self.activate_derivative(self.activations[i], self.hidden_activation)
            deltas.append(delta)
        
        # Reverse the order of deltas
        deltas.reverse()
        
        # Update weights and biases with L2 regularization
        for i in range(self.num_layers - 1):
            self.weights[i] += (np.dot(self.activations[i].T, deltas[i]) * self.learning_rate) - (self.l2_lambda * self.weights[i] * self.learning_rate)
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    def train(self, inputs, targets, epochs=10000, verbose=True, validation_data=None):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        num_samples = inputs.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            inputs_shuffled = inputs[perm]
            targets_shuffled = targets[perm]

            for i in range(0, num_samples, self.batch_size):
                batch_inputs = inputs_shuffled[i:i + self.batch_size]
                batch_targets = targets_shuffled[i:i + self.batch_size]
                output = self.forward(batch_inputs, training=True)
                self.backward(batch_inputs, batch_targets)

            loss = np.mean(np.square(targets - self.forward(inputs, training=False))) + (self.l2_lambda * sum([np.sum(np.square(w)) for w in self.weights]))
            self.losses.append(loss)
            accuracy = self.accuracy(self.forward(inputs, training=False), targets)
            self.accuracies.append(accuracy)

            if validation_data is not None:
                val_inputs, val_targets = validation_data
                val_loss = np.mean(np.square(val_targets - self.forward(val_inputs, training=False)))
                self.val_losses.append(val_loss)
                val_accuracy = self.accuracy(self.forward(val_inputs, training=False), val_targets)
                self.val_accuracies.append(val_accuracy)

            if verbose and epoch % 1000 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch}/{epochs}: loss = {loss:.4f}, val_loss = {val_loss:.4f}, accuracy = {accuracy:.4f}, val_accuracy = {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

    def accuracy(self, predictions, targets):
        return np.mean((predictions > 0.5) == targets)

    def predict(self, inputs):
        return self.forward(inputs, training=False)

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (X - mean) / std

def cross_validate(model_class, layer_sizes, X, y, k=5, epochs=10000, learning_rate=0.1, hidden_activation='relu', output_activation='sigmoid', dropout_rate=0.0, l2_lambda=0.0, batch_size=1, seed=None):
    kf = KFold(n_splits=k)
    fold = 1
    losses = []
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
        model = model_class(layer_sizes=layer_sizes, learning_rate=learning_rate, hidden_activation=hidden_activation, output_activation=output_activation, dropout_rate=dropout_rate, l2_lambda=l2_lambda, batch_size=batch_size, seed=seed)
        model.train(X_train, y_train, epochs=epochs, verbose=False, validation_data=(X_test, y_test))
        
        predictions = model.predict(X_test)
        loss = np.mean(np.square(y_test - predictions))
        accuracy = model.accuracy(predictions, y_test)
        print(f"Fold {fold}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")
        losses.append(loss)
        accuracies.append(accuracy)
        fold += 1
    print(f"Average Loss: {np.mean(losses):.4f}, Average Accuracy: {np.mean(accuracies):.4f}")
    return losses, accuracies

def hyperparameter_tuning(model_class, layer_sizes, X, y, param_grid, k=5, epochs=10000):
    best_loss = float('inf')
    best_params = None
    best_model = None
    all_results = []

    for params in ParameterGrid(param_grid):
        print(f"Testing with parameters: {params}")
        losses, accuracies = cross_validate(model_class, layer_sizes, X, y, k=k, epochs=epochs, **params)
        avg_loss = np.mean(losses)
        all_results.append((params, avg_loss, np.mean(accuracies)))
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params
            best_model = model_class(layer_sizes=layer_sizes, **params)

    print(f"Best Params: {best_params}, Best Loss: {best_loss:.4f}")
    return best_model, best_params, all_results

def plot_losses(model, filename="training_loss.png"):
    plt.plot(model.losses, label='Training Loss')
    if hasattr(model, 'val_losses'):
        plt.plot(model.val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('NeuralNetworks/' + filename)
    plt.close()

def plot_accuracies(model, filename="training_accuracy.png"):
    plt.plot(model.accuracies, label='Training Accuracy')
    if hasattr(model, 'val_accuracies'):
        plt.plot(model.val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.savefig('NeuralNetworks/' + filename)
    plt.close()

# Example usage
if __name__ == "__main__":
    # XOR problem
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Normalize the inputs
    inputs = normalize_data(inputs)

    # Define the network with 2 hidden layers
    layer_sizes = [2, 3, 2, 1]  # input layer, 2 hidden layers, output layer
    nn = NeuralNetwork(layer_sizes=layer_sizes, hidden_activation='relu', output_activation='sigmoid', learning_rate=0.1, seed=83, dropout_rate=0.2, l2_lambda=0.01, batch_size=4)
    nn.train(inputs, targets, epochs=10000, validation_data=(inputs, targets))

    # Predictions
    predictions = nn.predict(inputs)
    print("Predictions:")
    print(predictions)

    # Plot training and validation loss
    plot_losses(nn)

    # Plot training and validation accuracy
    plot_accuracies(nn)

    # Example usage with XOR problem and cross-validation
    cross_validate(NeuralNetwork, [2, 3, 2, 1], inputs, targets, k=3, epochs=10000, learning_rate=0.1, hidden_activation='relu', output_activation='sigmoid', dropout_rate=0.2, l2_lambda=0.01, batch_size=4)

    # Hyperparameter tuning example
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'hidden_activation': ['relu', 'sigmoid'],
        'output_activation': ['sigmoid'],
        'dropout_rate': [0.0, 0.2],
        'l2_lambda': [0.0, 0.01],
        'batch_size': [1, 4]
    }
    best_model, best_params, all_results = hyperparameter_tuning(NeuralNetwork, [2, 3, 2, 1], inputs, targets, param_grid, k=3, epochs=10000)

    # Train the model with the best parameters
    nn_best = NeuralNetwork(layer_sizes=layer_sizes, **best_params)
    nn_best.train(inputs, targets, epochs=10000, validation_data=(inputs, targets))

    # Predictions with the best model
    best_predictions = nn_best.predict(inputs)
    print("Best Model Predictions:")
    print(best_predictions)

    # Plot training and validation loss for the best model
    plot_losses(nn_best, filename="best_model_training_loss.png")

    # Plot training and validation accuracy for the best model
    plot_accuracies(nn_best, filename="best_model_training_accuracy.png")
