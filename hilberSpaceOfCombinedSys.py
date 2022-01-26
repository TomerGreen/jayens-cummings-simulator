import numpy as np
from scipy.linalg import logm
from qutip import *
from typing import Iterable
import matplotlib.pyplot as plt


N = 80  # max num of photons is N-1
ALPHA = 7
WC = 6 * 2 * np.pi * 1e9  # cavity frequency
G = 15 * 2 * np.pi * 1e6  # coupling strength

CASE_1 = {"wa": 6 * 2 * np.pi * 1e9, "tot_time": 1.6e-6, "steps": 8000, "dt": 0.2e-9}
CASE_2 = {"wa": 5 * 2 * np.pi * 1e9, "tot_time": 4e-6, "steps": 4000, "dt": 1e-9}


def jc_ham(case_params) -> Qobj:
    """Returns the Jaynes-Cummings Hamiltonian."""
    ham = WC * tensor(create(N) * destroy(N), identity(2))
    ham += (1 / 2) * case_params["wa"] * tensor(identity(N), sigmaz())
    ham -= G * (tensor(destroy(N), sigmap()) + tensor(create(N), sigmam()))
    return ham


def evolve_state(init_state: Qobj, ham: Qobj, case_params) -> list:
    """
    Evolves an initial state in time according to a given Hamiltonian and returns an array describing the state
    after each time step.
    :return An list of ket states, where each ket state is the time evolution of the initial state after the
    corresponding time step.
    """
    time_evo_op = Qobj.expm((-1j * case_params["dt"]) * ham)
    psi = init_state
    psi_t = list()
    for step in range(case_params["steps"]):
        psi = time_evo_op * psi
        psi_t.append(psi)
    return psi_t


def case1_init_state(alpha=ALPHA) -> Qobj:
    """Returns the initial combined state for case I"""
    bos_vac = basis(N)
    disp_op = Qobj.expm(alpha * create(N) - alpha.conjugate() * destroy(N))
    bos_coher = disp_op * bos_vac
    tls_gs = sigmam() * basis(2)
    psi0 = tensor(bos_coher, tls_gs)
    return psi0


def case2_init_state(alpha=ALPHA) -> Qobj:
    bos_vac = basis(N)
    disp_op = Qobj.expm(alpha * create(N) - alpha.conjugate() * destroy(N))
    bos_coher = disp_op * bos_vac
    tls_sup = (1/np.sqrt(2)) * (sigmam() * basis(2) + basis(2))
    psi0 = tensor(bos_coher, tls_sup)
    return psi0


def rdm(rho: Qobj) -> Qobj:
    """
    Returns the reduced density matrix of the two level system.
    :param rho: The combined state density matrix as a (2N, 2N) array.
    :return: The reduced density matrix as a (2,2) array.
    """
    rho_data = rho.data.toarray()
    rdm = np.empty((2, 2), np.complex128)
    for i, j in np.ndindex(rdm.shape):
        rho_part = rho_data[i::2, j::2]
        rdm[i, j] = np.trace(rho_part)
    return Qobj(rdm)


def draw_excited_state_population(psi_t: Iterable, case_params):
    """Draws the excited state population figure."""
    rdms = [rdm(psi * psi.dag()) for psi in psi_t]
    ee_pop = [rho[0, 0] for rho in rdms]
    times = np.linspace(0.0, case_params["steps"] * case_params["dt"], case_params["steps"])
    plt.plot(times, ee_pop, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Excited TLS Population")
    plt.suptitle(r'Two-Level System Excited State Population for $\alpha=$' + str(ALPHA))
    plt.show()


def draw_purity_plot(psi_t: Iterable, case_params):
    """Draws the purity figure"""
    rdms = [rdm(psi * psi.dag()).data.toarray() for psi in psi_t]
    purities = [np.trace(np.dot(rdm, rdm)) for rdm in rdms]
    times = np.linspace(0.0, case_params["steps"] * case_params["dt"], case_params["steps"])
    plt.plot(times, purities, linewidth=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel(r'$Tr(\rho_{TLS}^2)$')
    plt.suptitle(r'Purity of Combined State for $\alpha=$' + str(ALPHA))
    plt.show()


def draw_bosonic_state_density(psi_t: Iterable, case_params):
    """Draws state intensity figure"""
    psi = np.array(psi_t).squeeze()
    state_dens = np.array([np.real(st * np.conjugate(st)) for st in psi]).T
    state_dens = np.concatenate((state_dens[::2, :], state_dens[1::2, :]), axis=0)
    times = np.linspace(0.0, case_params["steps"] * case_params["dt"], case_params["steps"])
    times = times * np.ones_like(state_dens)
    states = np.expand_dims(np.arange(0, len(state_dens)), 1) * np.ones_like(state_dens)
    plt.pcolormesh(times, states, state_dens, cmap='inferno')
    plt.suptitle("Occupation Vs. Time")
    plt.ylabel("State Index")
    plt.xlabel("Time (s)")
    plt.show()


if __name__ == '__main__':
    params = CASE_1
    H = jc_ham(params)
    psi0 = case1_init_state()
    psi_t = evolve_state(psi0, H, params)
    draw_excited_state_population(psi_t, params)
    draw_purity_plot(psi_t, params)
    draw_bosonic_state_density(psi_t, params)

    params = CASE_2
    H = jc_ham(params)
    psi0 = case2_init_state()
    psi_t = evolve_state(psi0, H, params)
    draw_excited_state_population(psi_t, params)
    draw_purity_plot(psi_t, params)
    draw_bosonic_state_density(psi_t, params)
