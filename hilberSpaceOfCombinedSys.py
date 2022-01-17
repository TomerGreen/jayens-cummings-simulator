import numpy as np
from typing import Callable
from scipy.linalg import expm
from scipy.constants import hbar
# from qutip import *


N = 2  # number of quantum systems
# D = np.empty(2,10)
n = 100  # max num of photons
ALPHA = 7


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.identity(2)
PLUS_Y = np.array([[1, 1j], [1j, -1]]) / 2
MINUS_Y = np.array([[1, -1j], [-1j, -1]]) / 2
CONTROLLED_Y = np.kron(I, PLUS_Y) + np.kron(Z, MINUS_Y)

wc = 6 * 2 * np.pi * 10e9  # cavity frequency
wa = 6 * 2 * np.pi * 10e9  # atom frequency
g = 15 * 2 * np.pi * 10e6   # coupling strength
dt = 0.2 * 10e-9

bos_states = np.zeros((n, 1))  # initial state
bos_states[0, 0] = 1
TLS_states = np.array([[0, 1]]).T  #
psi0 = np.kron(bos_states, TLS_states)

a = np.zeros((n, n))
for i in range(1, n):
    a[i-1, i] = 1/np.sqrt(i)

a_dag = a.conj().T

sm = (X - 1j * Y) / 2
sm_dag = (X + 1j * Y) / 2

H = hbar * wc * np.kron(np.dot(a_dag, a), np.identity(2)) + hbar * np.kron(np.identity(n), wa * Z) - hbar * g * (np.kron(a, sm_dag) + np.kron(a_dag, sm))

evo_op = expm(H*(-1j)*dt / hbar)
disp_op = expm(ALPHA * a_dag - ALPHA.conjugate() * a)
coher_state = np.dot(disp_op, bos_states)
# print(coher_state)
print(np.linalg.norm(np.dot(a, coher_state)))
