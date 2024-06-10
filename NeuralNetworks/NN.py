import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation='sigmoid', output_activation='sigmoid', learning_rate=0.01, seed=None, dropout_rate=0.0, l2_lambda=0.0, batch_size=1):
        """
        Initializes a NeuralNetwork object.

        Args:
            layer_sizes (List[int]): A list of integers representing the number of neurons in each layer.
            hidden_activation (str, optional): The activation function to be used in the hidden layers. Defaults to 'sigmoid'.
            output_activation (str, optional): The activation function to be used in the output layer. Defaults to 'sigmoid'.
            learning_rate (float, optional): The learning rate for the optimization algorithm. Defaults to 0.01.
            seed (int, optional): The seed value for random number generation. Defaults to None.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.0.
            l2_lambda (float, optional): The L2 regularization parameter. Defaults to 0.0.
            batch_size (int, optional): The number of samples in each batch. Defaults to 1.

        Returns:
            None
        """
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
        """
        Calculates the sigmoid function for the given input.

        Parameters:
            x (float or array-like): The input value or values for which the sigmoid function is to be calculated.

        Returns:
            float or array-like: The sigmoid value or values corresponding to the input.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Calculates the derivative of the sigmoid function for the given input.

        Parameters:
            x (float or array-like): The input value or values for which the sigmoid derivative is to be calculated.

        Returns:
            float or array-like: The sigmoid derivative value or values corresponding to the input.
        """
        return x * (1 - x)

    def relu(self, x):
        """
        Apply the rectified linear unit (ReLU) activation function to the input.

        Parameters:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The output array after applying the ReLU activation function.
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Calculate the derivative of the Rectified Linear Unit (ReLU) activation function.

        Parameters:
            x (array-like): The input value or values for which the derivative is to be calculated.

        Returns:
            array-like: The derivative value or values corresponding to the input.
        """
        return np.where(x > 0, 1, 0)

    def activate(self, x, activation):
        """
        Apply the specified activation function to the input.

        Parameters:
            x (array-like): The input value or values for which the activation function is to be applied.
            activation (str): The name of the activation function to be used. Supported values are 'sigmoid' and 'relu'.

        Returns:
            array-like: The output value or values after applying the activation function.

        Raises:
            ValueError: If an unsupported activation function is specified.
        """

        if activation == 'sigmoid':
            return self.sigmoid(x)
        elif activation == 'relu':
            return self.relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def activate_derivative(self, x, activation):
        """
        Calculate the derivative of the specified activation function.

        Parameters:
            x (array-like): The input value or values for which the derivative is to be calculated.
            activation (str): The name of the activation function. Supported values are 'sigmoid' and 'relu'.

        Returns:
            array-like: The derivative value or values corresponding to the input.

        Raises:
            ValueError: If an unsupported activation function is specified.
        """

        if activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif activation == 'relu':
            return self.relu_derivative(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, inputs, training=True):
        """
        Performs the forward pass through the neural network.
        Args:
            inputs (ndarray): The input data of shape (batch_size, input_dim).
            training (bool, optional): Whether the forward pass is for training or inference. Defaults to True.
        Returns:
            ndarray: The output of the neural network of shape (batch_size, output_dim).
        """
        
        self.activations = [inputs]
        self.z_values = []

        for i in range(self.num_layers - 2):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.activate(z, self.hidden_activation)
            if activation is None:
                raise ValueError("Activation function returned None")
            if training and self.dropout_rate > 0:
                activation = self.dropout(activation, self.dropout_rate)
            self.activations.append(activation)

        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.activate(z, self.output_activation)
        if output is None:
            raise ValueError("Output activation function returned None")
        self.activations.append(output)

        return self.activations[-1]

    def dropout(self, x, rate):
        """
        Apply dropout to the input tensor.

        Args:
            x (numpy.ndarray): The input tensor.
            rate (float): The dropout rate.

        Returns:
            numpy.ndarray: The input tensor with dropout applied.
        """
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask

    def backward(self, inputs, targets):
        """
        Performs the backpropagation algorithm to update the weights and biases of the neural network.

        Args:
            inputs (ndarray): The input data of shape (batch_size, input_dim).
            targets (ndarray): The target values of shape (batch_size, output_dim).

        Returns:
            None
        """
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
        """
        Trains the neural network model on the given input data and targets.

        Args:
            inputs (ndarray): The input data of shape (batch_size, input_dim).
            targets (ndarray): The target values of shape (batch_size, output_dim).
            epochs (int, optional): The number of training epochs. Defaults to 10000.
            verbose (bool, optional): Whether to print the training progress. Defaults to True.
            validation_data (tuple, optional): The validation data in the form of (inputs, targets). Defaults to None.

        Returns:
            None
        """
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
        """
        Calculates the accuracy of the model's predictions.

        Parameters:
            predictions (ndarray): The predicted values.
            targets (ndarray): The ground truth values.

        Returns:
            ndarray: The accuracy of the model's predictions.
        """

        return np.mean((predictions > 0.5) == targets)

    def predict(self, inputs):
        """
        Perform a forward pass on the neural network with the given inputs.

        Parameters:
            inputs (ndarray): The input data to the neural network.

        Returns:
            ndarray: The output of the neural network.
        """
        return self.forward(inputs, training=False)

def normalize_data(X):
    """
    Normalizes the input data by subtracting the mean and dividing by the standard deviation.

    Parameters:
        X (ndarray): The input data to be normalized.

    Returns:
        ndarray: The normalized data with the same shape as the input.

    Note:
        If the standard deviation of a feature is zero, it is set to 1 to prevent division by zero.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (X - mean) / std

def cross_validate(model_class, layer_sizes, X, y, k=5, epochs=10000, learning_rate=0.1, hidden_activation='relu', output_activation='sigmoid', dropout_rate=0.0, l2_lambda=0.0, batch_size=1, seed=None):
    """
    Cross-validates a given model class using k-fold validation.

    Args:
        model_class (class): The class of the model to be cross-validated.
        layer_sizes (list): A list of integers representing the number of neurons in each layer of the model.
        X (ndarray): The input data for training and testing.
        y (ndarray): The target values for training and testing.
        k (int, optional): The number of folds in k-fold cross-validation. Defaults to 5.
        epochs (int, optional): The number of training epochs. Defaults to 10000.
        learning_rate (float, optional): The learning rate for the model. Defaults to 0.1.
        hidden_activation (str, optional): The activation function for the hidden layers. Defaults to 'relu'.
        output_activation (str, optional): The activation function for the output layer. Defaults to 'sigmoid'.
        dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.0.
        l2_lambda (float, optional): The L2 regularization parameter. Defaults to 0.0.
        batch_size (int, optional): The number of samples in each batch. Defaults to 1.
        seed (int, optional): The seed value for random number generation. Defaults to None.

    Returns:
        tuple: A tuple containing the average loss and accuracy across all folds.
    """    
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
    """
    Hyperparameter tuning for a given model class using k-fold cross-validation.

    Args:
        model_class (class): The class of the model to be tuned.
        layer_sizes (list): A list of integers representing the number of neurons in each layer of the model.
        X (ndarray): The input data for training and testing.
        y (ndarray): The target values for training and testing.
        param_grid (dict): A dictionary of parameter grids to test.
        k (int, optional): The number of folds in k-fold cross-validation. Defaults to 5.
        epochs (int, optional): The number of training epochs. Defaults to 10000.

    Returns:
        tuple: A tuple containing the best model, best parameters, and a list of all results.

    """    
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
    """
    Plot the training and validation losses over epochs.

    Args:
        model (object): The model object containing the losses.
        filename (str, optional): The name of the file to save the plot. Defaults to "training_loss.png".

    Returns:
        None

    This function plots the training and validation losses over epochs using matplotlib. The training loss is plotted with the label 'Training Loss', and if the model object has a 'val_losses' attribute, the validation loss is also plotted with the label 'Validation Loss'. The x-axis represents the epochs, and the y-axis represents the loss. The plot is titled 'Training and Validation Loss Over Epochs'. The legend is displayed to differentiate between the training and validation losses. The plot is saved as an image file with the specified filename in the 'NeuralNetworks' directory. The plot is then closed to free up memory.
    """
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
    """
    Plot the training and validation accuracies over epochs.

    Args:
        model (object): The model object containing the accuracies.
        filename (str, optional): The name of the file to save the plot. Defaults to "training_accuracy.png".

    Returns:
        None

    This function plots the training and validation accuracies over epochs using matplotlib. The training accuracy is plotted with the label 'Training Accuracy', and if the model object has a 'val_accuracies' attribute, the validation accuracy is also plotted with the label 'Validation Accuracy'. The x-axis represents the epochs, and the y-axis represents the accuracy. The plot is titled 'Training and Validation Accuracy Over Epochs'. The legend is displayed to differentiate between the training and validation accuracies. The plot is saved as an image file with the specified filename in the 'NeuralNetworks' directory. The plot is then closed to free up memory.
    """
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
