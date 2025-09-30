import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def build_alpha_circuit(theta: float):
    """
    Build a simple 2-qubit 'alpha' toy circuit:
    - Entangle qubits (representing  bound nucleon pair behavior)
    - Apply symmetric rotation (excitation) controlled by theta
    - Save statevector for inspection and measurement
    """

    qc = QuantumCircuit(2, 2)

    #Create entangelment representing binding
    qc.h(0)
    qc.cx(0, 1)

    #Symmetric excitation rotations (toy model)
    qc.ry(theta, 0)
    qc.ry(theta, 1)

    #attach save_statevector instruction so simulator returns the statevector
    qc.save_statevector()

    #Also add measurement for statistics (we'll run a separate circuit for shots)
    qcMeas = qc.copy()
    qcMeas.measure([0, 1], [0, 1])

    return qc, qcMeas


def run_simulation(theta: float, shots: int=1024):
    """
    Runs the circuit and returns (statevector_np, counts)
    - Statevector_np: numpy array of complex amplitudes (length 4 for 2 qubits)
    - counts: measurement counts dict (strings -> ints)
    """
    qcSv, qcMeas = build_alpha_circuit(theta)
    sim = AerSimulator()
    
    #run statevector circuit
    tQc = transpile(qcSv, sim)
    job = sim.run(tQc)
    res = job.result()

    #get statevector as numpy array (explicitly cast)
    sv = res.get_statevector(tQc)   # returns qiskit.quantum_info.statevector
    svNp = np.asarray(sv)   #convert to numpy array (complex)

    #run measurement circuit for counts
    tMeas = transpile(qcMeas, sim)
    job2 = sim.run(tMeas, shots=shots, seed_simulator=None)
    res2 = job2.result()
    counts = res2.get_counts(tMeas)

    return svNp, counts
