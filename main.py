import numpy as np
from simulator import run_simulation
from viz import map_state_to_displacements, plot_alpha, plot_probs_bar
from aiModel import build_and_train
import matplotlib.pyplot as plt

def run_demo():
    print(f"Generating dataset and training AI (this may take a minute)...")
    model, X, y = build_and_train(nSample=48)
    print(f"Training complete. Example training MSE: {((model.predict(X)-y)**2).mean()}")

    #Pick a random theta to simulate
    theta = float(np.random.uniform(0, np.pi))
    print(f"\nSimulating quantum alpha for theta = {theta:.3f} rad")
    sv, counts = run_simulation(theta, shots=1024)
    probs = np.abs(sv)**2
    print(f"Measured counts (approx): {counts}\nState Probabilities: {np.round(probs, 4)}")

    #Map to 3D positions and show
    positions, probs = map_state_to_displacements(sv, scale=0.9)
    plot_alpha(positions, probs, title=f"Alpha particle (theta={theta:.3f})")
    plot_probs_bar(probs)

    #AI prediction of probs for same theta
    yPred = model.predict(np.array([[theta]]))[0]
    print(f"AI predicted probs: {np.round(yPred, 4)}")

    #Visualize AI-Predicted structure (reconstruct positions by using predicted probs)
    #Create pesudo-statevector with magnitudes sqrt(predProbs) and zero phases
    predSv = np.sqrt(np.clip(yPred, 0, 1)).astype(complex)
    predPositions, predProbs = map_state_to_displacements(predSv, scale=0.9)
    plot_alpha(predPositions, predProbs, title=f"AI Predicted Structure (theta={theta:.3f})")
    plot_probs_bar(predProbs)


if __name__ == "__main__":
    run_demo()