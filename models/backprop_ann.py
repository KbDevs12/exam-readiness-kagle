import numpy as np


class BackpropANN:
    """ANN Backpropagation sederhana dari nol untuk regresi exam_score."""

    def __init__(self, input_size, hidden_size=10, output_size=1, seed=7):
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train(self, X, y, epochs=1800, lr=0.08):
        history_loss = []
        for _ in range(epochs):
            predicted = self.forward(X)
            error = y - predicted
            history_loss.append(float(np.mean(error ** 2)))

            d_output = error * self.sigmoid_derivative(predicted)
            error_hidden = d_output.dot(self.w2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

            self.w2 += self.a1.T.dot(d_output) * lr / len(X)
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr / len(X)
            self.w1 += X.T.dot(d_hidden) * lr / len(X)
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr / len(X)
        return history_loss

    def predict(self, X):
        return self.forward(X)
