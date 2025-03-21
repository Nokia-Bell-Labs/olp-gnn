'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import numpy as np
from math import sqrt, log10
import cvxpy as cp
import time
from pypapi import papi_high


# Function that solves the SOCP problem for given parameters
# (A, B, G, rho_d, G_norm) define the problem's constraints
# solver is the solver name
def opti_mr(t, A, B, solver, verbose, G, rho_d, G_norm, M, K):
    # parameter to optimize, power control coefficients
    ksi = cp.Variable((M*K+1, 1), pos=True)

    ksi_mat = cp.reshape(ksi[0:M*K, 0], (K, M)).T
    constraints = [ksi[M*K, 0] == 1]  # set the last value of ksi
    for k in range(K):  # SINR constraint, SOCP problem
        b_k = np.reshape(B[k], (1, M))
        A_k = A[k]
        constraints = constraints + \
            [b_k @ cp.reshape(ksi[k*M:(k+1)*M], (M, 1)) >=
             sqrt(t)*cp.pnorm(A_k@ksi, p=2)]
    for m in range(K*M):  # positivity constraint
        constraints += [ksi[m, 0] >= 0]
    for m in range(M):  # power constraints keep sum<1
        constraints += [
            cp.pnorm(cp.multiply(cp.reshape(ksi[0:M*K, 0], (K, M)).T,
                                 G_norm/sqrt(rho_d))[m, :], 2) <= 1,
            cp.pnorm(ksi_mat[m, :], 2) <= sqrt(rho_d)]
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose)
    return prob, ksi


# Bisection search to find the optimal SINR and Delta matrix
# (low,up,eps) parameters of the bisection search
def MR_solver(low, up, eps, solver, channel_gen, M, K, papi_events):
    if papi_events is not None:
        papi_high.start_counters(papi_events)
    start_time = time.process_time()
    B = []
    A = []
    G, rho_d = channel_gen(M, K)
    G_norm = np.conjugate(G)*(1/np.abs(G))

    ite = 0
    for k in range(K):
        G_k = np.repeat(np.reshape(G[:, k], (M, 1)), K, axis=1)  # format M*K
        # a_k=(a_k1...a_kl...a_kK), dim(a_kl)=column
        a_k = np.conjugate(G_k * G_norm)  # matrix of (g_kl*)/abs(g_kl)
        tmp_A_k = np.concatenate(
            (np.reshape(np.conjugate(a_k.T), (1, M*K)), np.ones((1, 1))),
            axis=1)
        A_k = np.zeros((K+1, M*K+1), dtype=complex)
        A_k[K, M*K] = 1
        B.append(np.reshape(np.abs(G[:, k]), (M, 1)))
        for i in range(K):  # sinr constraint
            A_k[i, i*M:(i+1)*M] = tmp_A_k[0, i*M:(i+1)*M]
            A_k[k, :] = 0
        A.append(A_k)
    ksi_test = np.zeros((M*K+1, 1))
    ksi_opt = np.zeros((M*K+1, 1))
    lowb = min(low, up)
    upb = max(low, up)
    while abs(lowb-upb) > eps:
        ite += 1
        tSINR = (lowb+upb) / 2
        try:
            prob, ksi_test = opti_mr(
                tSINR, A, B, solver, False, G, rho_d, G_norm, M, K)
            is_feasible = prob.value < np.inf
        except cp.SolverError:
            # print('Solver MOSEK status UNKNOWN')
            is_feasible = False

        if is_feasible:
            lowb = tSINR
            ksi_opt = ksi_test.value
            best_SINR = tSINR
        else:
            upb = tSINR

    eta_opt = (np.reshape(ksi_opt[:M*K, 0], (K, M)
                          ).T/sqrt(rho_d))**2
    Delta_opt = np.sqrt(eta_opt)*G_norm
    if papi_events is not None:
        flops = papi_high.stop_counters()
    else:
        flops = [-1]
    end_time = time.process_time()
    time_total = end_time-start_time
    return [best_SINR, Delta_opt, time_total, ite, G, flops[0]]


# Function that generates and solves n MR problems of size M APs and K UEs
# The solver used here is MOSEK
# channel_gen: generates the channel matrix given M and K
# papi_events: is passed to count the FLOPs of the solver
def data_generation_mr(n, channel_gen, M, K, papi_events=None, verbose=True):
    SINR = []
    Delta = []
    G = []
    flops = []
    low, up = 0, 10**6
    eps = 0.1

    for i in range(n):
        sol = MR_solver(low, up, eps, 'MOSEK', channel_gen, M, K, papi_events)
        cur_SINR = 10*log10(sol[0])  # from linear to dB
        SINR.append(cur_SINR)
        Delta.append(sol[1])
        G.append(sol[4])
        flops.append(sol[5])
        if verbose:
            print(('MR sample number {} done in {:.2e}s and '
                   '{} steps, FLOPS={:.2e}, SINR={:.2e}dB')
                  .format(i, sol[2], sol[3], sol[5], cur_SINR))

    out_dict = {'SINR': SINR, 'Delta': Delta, 'G': G, 'flops': flops}
    return out_dict
