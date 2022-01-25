import numpy as np
from scipy.linalg import expm
from scipy.constants import hbar


#  ==== Params ==== #
N = 10  # max num of photons is N-1
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
    """Returns the Jaynes-Cummings Hamiltonian as a (2N,2N) array."""
    ham = hbar * wc * np.kron(np.dot(A_DAG, A), np.identity(2)) + hbar * wa * np.kron(np.identity(N), Z)
    ham -= hbar * g * (np.kron(A, SP) + np.kron(A_DAG, SM))
    return ham


def evolve_state(init_state: np.ndarray, ham: np.ndarray, steps, dt=DT):
    """
    Evolves an initial state in time according to a given Hamiltonian and returns an array describing the state
    after each time step.
    :param init_state: The initial state as an (2N,1) array.
    :param ham: The Hamiltonian as a (2N,2N) array.
    :param steps: The number of time steps.
    :param dt: The duration of time step.
    :return An array of shape (2N, steps) where each column contains the amplitudes of the state after the
    corresponding time step.
    """
    time_evo_op = expm((-1j * dt / hbar) * ham)
    psi = init_state
    psi_t = np.zeros(psi.shape[0], steps)
    for step in range(steps):
        psi = time_evo_op.dot(psi)
        psi_t[:, steps] = psi
    return psi_t


def disp_op(alpha=ALPHA):
    """Returns a bosonic coherent state as an (N,N) array."""
    return expm(alpha * A_DAG - ALPHA.conjugate() * A)


def get_init_state(alpha=ALPHA):
    """Returns the initial state as an (2N,1) array."""
    bos_vac = np.zeros((N, 1))  # initial bosonic state
    bos_vac[0, 0] = 1
    bos_coher = disp_op(alpha).dot(bos_vac)
    tls_state = np.array([[0, 1]]).T  # initial TLS state
    psi0 = np.kron(bos_coher, tls_state)
    return psi0


if __name__ == '__main__':
    psi0 = get_init_state()
