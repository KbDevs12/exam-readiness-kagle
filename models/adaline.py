import numpy as np


class Adaline:
    """Adaline untuk klasifikasi high risk berdasarkan performa ujian rendah."""

    def __init__(self, input_size, seed=11):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-0.05, 0.05, input_size)
        self.bias = 0.0

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, epochs=1200, lr=0.02):
        history_loss = []
        y = np.asarray(y, dtype=float)
        for _ in range(epochs):
            output = self.net_input(X)
            error = y - output
            self.weights += lr * X.T.dot(error) / len(X)
            self.bias += lr * np.mean(error)
            history_loss.append(float(np.mean(error ** 2)))
        return history_loss

    def predict_score(self, X):
        raw = self.net_input(X)
        return 1 / (1 + np.exp(-raw))

    def predict_label(self, X, threshold=0.5):
        return (self.predict_score(X) >= threshold).astype(int)
