import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#base tetrahedral positions(centered)
BASE_POSITIONS = np.array([
    [ 1.0,  1.0,  1.0],   # nucleon 0 (proton)
    [-1.0, -1.0,  1.0],   # nucleon 1 (proton)
    [-1.0,  1.0, -1.0],   # nucleon 2 (neutron)
    [ 1.0, -1.0, -1.0],   # nucleon 3 (neutron)
], dtype=float)

LABELS = ["proton", "proton","neutron", "neutron"]
COLORS = ["red", "red", "gray", "gray"]


def map_state_to_displacements(statevector: np.ndarray, scale: float=0.6):
    """
    Map a 4-element statevector to displacements for 4 nucleons.
    This toy mapping:
    - Compute probabilities p_i = |a_i|^2 for the4 computational basis states
    - Center probabilities (subtract mean) and scale them
    - displace each nucleon  radially from center by disp_i = scale * (p_i - mean_p)
    Return new_positions (4, 3) with base + displacement vectors.
    """

    probs = np.abs(statevector)**2

    #ensure length 4
    assert probs.shape[0] == 4
    mean_p = np.mean(probs)

    #radial vectors from center
    centers = BASE_POSITIONS.copy()
    radii = np.linalg.norm(centers, axis=1)

    #normalized radial directions
    dirs = centers / np.maximum(radii.reshape(-1, 1), 1e-8)
    displacements = ((probs - mean_p).reshape(-1, 1)) * scale* dirs
    positions = BASE_POSITIONS + displacements
    return positions, probs


def plot_alpha(positions, probs, title="Alpha Particle (quantum -> 3D)"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    #Plot spheres (scatter) with color and size by probability
    for i, pos in enumerate(positions):
        ax.scatter(pos[0], pos[1], pos[2], color=COLORS[i], s=300 *(0.2 + probs[i]), edgecolors='k', alpha=0.9, label=LABELS[i] if i<2 else None)
        ax.text(pos[0]*1.05, pos[1]*1.05, pos[2]*1.05 , f"{i} p={probs[i]:.2f}", fontsize=9)

    #Connect nucleons with lines to show cluster shape
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            xs = [positions[i, 0], positions[j, 0]]
            ys = [positions[i, 1], positions[j, 1]]
            zs = [positions[i, 2], positions[j, 2]]
            ax.plot(xs, ys, zs, color='k', linewidth=0.6, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-2,2], [-2,2], [-2,2])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def plot_probs_bar(probs):
    plt.figure(figsize=(6, 3))
    labels = ["|00⟩","|01⟩","|10⟩","|11⟩"]
    plt.bar(labels, probs, color=['C0','C1','C2','C3'])
    plt.ylim(0, 1)
    plt.title("Basis state probabilities")
    plt.ylabel("probability")
    plt.show()
