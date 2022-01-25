import numpy as np
from scipy.linalg import logm
from qutip import *
from typing import Iterable
import matplotlib.pyplot as plt


N = 100  # max num of photons is N-1
ALPHA = 7
WC = 6 * 2 * np.pi * 1e9  # cavity frequency
WA = 6 * 2 * np.pi * 1e9  # atom frequency
G = 15 * 2 * np.pi * 1e6   # coupling strength
DT = 0.2e-9
TOT_TIME = 1.6e-6
STEPS = 8000


def jc_ham(wc=WC, wa=WA, g=G) -> Qobj:
    """Returns the Jaynes-Cummings Hamiltonian."""
    ham = wc * tensor(create(N) * destroy(N), identity(2))
    ham += (1 / 2) * wa * tensor(identity(N), sigmaz())
    ham -= g * (tensor(destroy(N), sigmap()) + tensor(create(N), sigmam()))
    return ham


def evolve_state(init_state: Qobj, ham: Qobj, steps, dt=DT) -> list:
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
        psi = time_evo_op * psi
        psi_t.append(psi)
    return psi_t


def init_state(alpha=ALPHA) -> Qobj:
    """Returns the initial two-system state"""
    bos_vac = basis(N)
    disp_op = Qobj.expm(alpha * create(N) - alpha.conjugate() * destroy(N))
    bos_coher = disp_op * bos_vac
    tls_gs = sigmam() * basis(2)
    psi0 = tensor(bos_coher, tls_gs)
    return psi0


def draw_excited_state_population(psi_t: Iterable, steps=STEPS, dt=DT):
    """Draws the excited state population figure."""
    rdms = [state.ptrace(1) for state in psi_t]
    ee_pop = [rho[0, 0] for rho in rdms]
    times = np.linspace(0.0, steps * dt, steps)
    plt.plot(times, ee_pop, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Excited TLS Population")
    plt.suptitle(r'Two-Level System Excited State Population for $\alpha=$' + str(ALPHA))
    plt.show()


def draw_purity_plot(psi_t: Iterable, steps=STEPS, dt=DT):
    """Draws the purity figure"""
    rdms = [state.ptrace(1).data.toarray() for state in psi_t]
    # purities = [1-np.trace(np.dot(rdm, logm(rdm))) for rdm in rdms]
    purities = [np.trace(np.dot(rdm, rdm)) for rdm in rdms]
    times = np.linspace(0.0, steps * dt, steps)
    plt.plot(times, purities, linewidth=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel(r'$Tr(\rho_{TLS}^2)$')
    plt.suptitle(r'Purity of Combined State for $\alpha=$' + str(ALPHA))
    plt.show()


if __name__ == '__main__':
    H = jc_ham()
    psi0 = init_state()
    psi_t = evolve_state(psi0, H, STEPS)
    draw_purity_plot(psi_t)
