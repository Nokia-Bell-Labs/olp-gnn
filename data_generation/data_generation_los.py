'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import os
import sys
import numpy as np
from channel_generation import create_channel_los
from olp_precoding import data_generation_olp
from zf_precoding import data_generation_zf
from mr_precoding import data_generation_mr
from pypapi import events


N = int(sys.argv[1])  # number of samples/graphs to generate
M = int(sys.argv[2])  # number of APs
K = int(sys.argv[3])  # number of UEs
f = int(sys.argv[4])  # frequency in GHz
f_hz = f * (10**9)    # frequency in Hz
precoder = sys.argv[5]  # precoding algorithm: olp, mr, zf

# If FLOPS count is not required, then set papi_events to None
papi_events = [events.PAPI_DP_OPS]  # None

assert f in [2, 60], "frequency must be either 2 or 60 (GHz)"
assert precoder in ['olp', 'mr', 'zf'], ("precoder must be one of \'olp\', "
                                         "\'mr\' and \'zf\'.")


def my_channel_generation(M, K):
    return create_channel_los(f_hz, M, K)


print(('Generating {} samples using {} for {} GHz LoS model with {} APs and '
       '{} UEs:').format(N, precoder, f, M, K))

if precoder == 'olp':
    data = data_generation_olp(N, my_channel_generation, M, K, papi_events)
elif precoder == 'zf':
    data = data_generation_zf(N, my_channel_generation, M, K, papi_events)
else:
    data = data_generation_mr(N, my_channel_generation, M, K, papi_events)

# Create 'data' folder if it does not already exists
os.makedirs('data', exist_ok=True)

basefilename = 'data/data_los_{}GHz_{}_{}_{}'.format(f, precoder, M, K)
if os.path.exists(basefilename+'.npz'):
    i = 1
    filename = "{}({})".format(basefilename, i)
    while os.path.exists(filename+'.npz'):
        i += 1
        filename = "{}({})".format(basefilename, i)
    np.savez(filename, **data)
else:
    np.savez(basefilename, **data)
