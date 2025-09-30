import numpy as np
from sklearn.neural_network import MLPRegressor
from simulator import run_simulation #local import (same package)

def generate_dataset(nSample=60):
    X = []
    y = []
    thetas = np.linspace(0, np.pi, nSample)
    for theta in thetas:
        sv, counts = run_simulation(theta, shots=512)
        probs = np.abs(sv)**2   #4-d vector

        #We'll predict the 4 probabilities as target
        X.append([theta])
        y.append(probs)
    return np.array(X), np.array(y)

def train_model(X, y):
    #Flatten target to shape (nSample, 4)
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    model.fit(X,y)
    return model

def build_and_train(nSample=60):
    X, y = generate_dataset(nSample=nSample)
    model = train_model(X, y)
    return model, X, y