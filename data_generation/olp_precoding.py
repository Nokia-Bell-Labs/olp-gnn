'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import numpy as np
from math import log10, sqrt
import cvxpy as cp
import time
from pypapi import papi_high


# SOCP problem solver
# (G_dague, P_G, rho_d) set the problem's constraints
# t: is the currently computed lower bound sinr
def opti_OLP(t, solver, G_dague, P_G, rho_d, M, K):
    A = cp.Variable(shape=(K, K), complex=True)
    A_diag = cp.Variable(shape=(K, 1), pos=True)
    A_tilde = cp.Variable(shape=(K, K+1), complex=True)
    constraints = [cp.reshape(A_tilde[:, K], (K, 1))
                   # keep the last column constant
                   == np.ones((K, 1))/sqrt(rho_d)]
    U = cp.Variable(shape=(M, K), complex=True)
    for i in range(K):
        for j in range(K):
            # create the link between A and A_tilde for non diag element
            if i == j:
                constraints += [A_tilde[i, j] == 0]
                constraints += [A[i, j] == A_diag[i, 0]]
            # can't set A_tilde==A directly because diag(A_tilde)==0 but not
            # diag(A)
            else:
                constraints += [A_tilde[i, j] == A[i, j]]
        constraints += [A_diag[i, 0] >= sqrt(t)*cp.pnorm(A_tilde[i, :], 2)]

    Delta = G_dague @ A + P_G @ U

    for m in range(M):
        constraints += [cp.pnorm(Delta[m, :], 2) <= 1]
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=False)

    return prob, A, U


# Function solving the B-SOCP by performing bisection search on top
# of opti_OLP (SOCP) calls
def OLP_solver(low, up, eps, solver, channel_gen, M, K, papi_events):
    start_time = time.process_time()
    if papi_events is not None:
        papi_high.start_counters(papi_events)
    G, rho_d = channel_gen(M, K)
    G_inv = np.linalg.inv((np.conjugate(G).T).dot(G))  # =(G*G)**-1
    G_dague = np.conjugate(G).dot(G_inv.T)
    P_G = np.eye(M) - G_dague.dot(G.T)

    U_opt = np.zeros((M, K))
    A_opt = np.zeros((K, K))
    U_test = np.zeros((M, K))
    A_test = np.zeros((K, K))
    lowb = min(low, up)
    upb = max(low, up)
    ite = 0
    best_SINR = 0.0
    while abs(lowb-upb) > eps:
        ite += 1
        tSINR = (lowb+upb) / 2
        try:
            prob, A_test, U_test = opti_OLP(
                tSINR, solver, G_dague, P_G, rho_d, M, K)
            is_feasible = prob.value < np.inf
        except cp.SolverError:
            # print('Solver MOSEK status UNKNOWN')
            is_feasible = False

        if is_feasible:
            lowb = tSINR
            A_opt, U_opt = A_test.value, U_test.value
            best_SINR = tSINR
        else:
            upb = tSINR

    Delta_opt = G_dague @ A_opt + P_G @ U_opt
    if papi_events is not None:
        flops = papi_high.stop_counters()
    else:
        flops = [-1]
    end_time = time.process_time()
    time_total = end_time-start_time
    return [best_SINR, Delta_opt, time_total, ite, G, flops[0], A_opt, U_opt]


# Function that generates and solves n OLP problems of size M APs and K UEs
# The solver used here is MOSEK
# channel_gen: generates the channel matrix given M and K
# papi_events: is passed to count the FLOPs of the solver
def data_generation_olp(n, channel_gen, M, K, papi_events=None, verbose=True):
    Delta = []
    SINR = []
    G = []
    flops = []
    A = []
    U = []
    low, up = 0, 10**6
    eps = 0.01

    for i in range(n):
        sol = OLP_solver(low, up, eps, 'MOSEK', channel_gen, M, K, papi_events)
        cur_SINR = 10*log10(sol[0])  # from linear to dB
        SINR.append(cur_SINR)
        Delta.append(sol[1])
        G.append(sol[4])
        flops.append(sol[5])
        A.append(sol[6])
        U.append(sol[7])
        if verbose:
            print(('OLP sample number {} done in {:.2e}s and '
                   '{} steps, FLOPS={:.2e}, SINR={:.2e}dB')
                  .format(i, sol[2], sol[3], sol[5], cur_SINR))

    out_dict = {'SINR': SINR, 'Delta': Delta, 'G': G, 'flops': flops,
                'A': A, 'U': U}
    return out_dict
