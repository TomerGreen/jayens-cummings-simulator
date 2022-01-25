import numpy as np
# from scipy.linalg import expm
from qutip import *


#  ==== Params ==== #
N = 5  # max num of photons is N-1
ALPHA = 7
WC = 6 * 2 * np.pi * 10e9  # cavity frequency
WA = 6 * 2 * np.pi * 10e9  # atom frequency
G = 15 * 2 * np.pi * 10e6   # coupling strength
DT = 0.2 * 10e-9


# === CONSTANTS === #
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
SM = (X - 1j * Y) / 2
SP = (X + 1j * Y) / 2
A = np.zeros((N, N))
for i in range(1, N):
    A[i - 1, i] = np.sqrt(i)
A_DAG = A.conj().T


def jc_ham(wc=WC, wa=WA, g=G):
    """Returns the Jaynes-Cummings Hamiltonian."""
    ham = wc * tensor(create(N) * destroy(N), identity(2))
    ham += (1 / 2) * wa * tensor(identity(N), sigmaz())
    ham -= g * (tensor(destroy(N), sigmap()) + tensor(create(N), sigmam()))
    return ham


def evolve_state(init_state: Qobj, ham: Qobj, steps, dt=DT):
    """
    Evolves an initial state in time according to a given Hamiltonian and returns an array describing the state
    after each time step.
    :return An list of ket states, where each ket state is the time evolution of the initial state after the
    corresponding time step.
    """
    time_evo_op = Qobj.expm((-1j * dt) * ham)
    psi = init_state
    psi_t = list()
    for step in range(steps):
        print(psi)
        psi = time_evo_op * psi
        psi_t.append(psi)
    return psi_t


def init_state(alpha=ALPHA):
    """Returns the initial two-system state"""
    bos_vac = basis(N)
    disp_op = Qobj.expm(alpha * create(N) - alpha.conjugate() * destroy(N))
    bos_coher = disp_op * bos_vac
    tls_gs = sigmam() * basis(2)
    psi0 = tensor(bos_coher, tls_gs)
    return psi0


if __name__ == '__main__':
    H = jc_ham()
    psi0 = init_state()
    psi_t = evolve_state(psi0, H, 8000, DT)
