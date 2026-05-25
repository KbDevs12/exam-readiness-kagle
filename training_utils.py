from models import BackpropANN


def compare_learning_rates(X, y, input_size, hidden_size=10, epochs=600):
    results = {}
    for lr in [0.01, 0.05, 0.08, 0.2]:
        model = BackpropANN(input_size=input_size, hidden_size=hidden_size, seed=21)
        results[lr] = model.train(X, y, epochs=epochs, lr=lr)
    return results
