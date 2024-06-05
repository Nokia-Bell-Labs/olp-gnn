'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import numpy as np
from math import sqrt, log10
import time
from pypapi import papi_high


# Computes the optimal zero-forcing precoder
def zeroforcing(channel_gen, M, K, papi_events):
    if papi_events is not None:
        papi_high.start_counters(papi_events)
    start_time = time.process_time()
    G, rho_d = channel_gen(M, K)
    G_inv = np.linalg.inv((np.conjugate(G).T).dot(G))
    G_dague = np.conjugate(G).dot(G_inv.T)

    eta_list = []
    for m in range(M):
        G_m = np.reshape(G[m, :], (K, 1))
        g_m = np.conjugate(G_m).dot(G_m.T)
        eta_list.append(np.trace(G_inv.dot(g_m.dot(G_inv))))
    eta = abs(1/max(eta_list))
    SINR_opt = eta*rho_d
    Eta_sqrt = np.diag(sqrt(abs(eta))*np.ones(K))
    end_time = time.process_time()
    time_total = end_time-start_time
    Eta_sqrt = np.diag(sqrt(abs(eta))*np.ones(K))
    Delta_zf = G_dague.dot(Eta_sqrt)
    if papi_events is not None:
        flops = papi_high.stop_counters()
    else:
        flops = [-1]
    return [SINR_opt, Delta_zf, time_total, G, flops[0]]


# Function that generates and solves n ZF problems of size M APs and K UEs
# channel_gen: generates the channel matrix given M and K
# papi_events: is passed to count the FLOPs of the solver
def data_generation_zf(n, channel_gen, M, K, papi_events=None, verbose=True):
    SINR = []
    Delta = []
    G = []
    flops = []
    for i in range(n):
        sol = zeroforcing(channel_gen, M, K, papi_events)
        SINR.append(sol[0])
        Delta.append(sol[1])
        G.append(sol[3])
        flops.append(sol[4])
        if verbose:
            print(('ZF sample number {} done in {:.2e}s'
                   ', FLOPS={:.2e}, SINR={:.2e}dB')
                  .format(i, sol[2], sol[4], sol[0]))

    for i in range(n):  # from linear to dB
        SINR[i] = 10*log10(SINR[i])

    out_dict = {'SINR': SINR, 'Delta': Delta, 'G': G, 'flops': flops}
    return out_dict
