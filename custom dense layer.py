import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        # He-style small random init (keeps gradients sane)
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.lr = learning_rate

    def forward(self, X):
        """
        X shape: (batch_size, input_dim)
        """
        self.X = X  # cache for backprop
        out = X @ self.W + self.b
        return out

    def backward(self, dL_dout):
        """
        dL_dout shape: (batch_size, output_dim)
        """
        # gradients
        dL_dW = self.X.T @ dL_dout
        dL_db = np.sum(dL_dout, axis=0, keepdims=True)
        dL_dX = dL_dout @ self.W.T

        # parameter update (SGD)
        self.W -= self.lr * dL_dW
        self.b -= self.lr * dL_db

        return dL_dX
