'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import numpy as np
from math import log10, pi, sqrt


def create_channel_nlos(mor, M, K):
    '''
    Create channel in non-line-of-sight (NLoS) propagation model

    Parameters
    ----------
    mor :   channel propagation morphology/scenario, can be one of 'urban',
            'suburban', 'rural'
    M :     number of APs
    K :     number of UEs

    Returns
    -------
    G :     complex channel coefficients matrix
    rho_d : downlink SNR (the power coefficients are normalized by rho_d to
            get a value between 0 and 1).
    '''

    assert mor in ['urban', 'suburban', 'rural'], ("mor must be one of "
                                                   "\'urban\', \'suburban\', "
                                                   "\'rural\'.")
    if mor == 'urban':
        R = 0.5  # cell radius, in km
        f = 2000  # carrier frequency in MHz
        W = 20  # street width, in meters,
        h = 20  # average building height, in meters
        hte = 20  # effective AP antenna height in meters
        hre = 1.5  # effective mobile antenna height in meters
        sfstd = 6  # slow fading standard deviation in dB
    elif mor == 'suburban':
        R = 1  # cell radius, in km
        f = 2000  # carrier frequency in MHz
        W = 20  # street width, in meters,
        h = 10  # average building height, in meters
        hte = 20  # effective AP antenna height in meters
        hre = 1.5  # effective mobile antenna height in meters
        sfstd = 8  # slow fading standard deviation in dB
    elif mor == 'rural':
        R = 4   # cell radius, in km
        f = 450  # carrier frequency in MHz
        W = 20  # street width, in meters,
        h = 5  # average building height, in meters
        hte = 40  # effective AP antenna height in meters
        hre = 1.5  # effective mobile antenna height in meters
        sfstd = 8  # slow fading standard deviation in dB

    B = 20  # bandwidth (MHz)
    APP = 0.2  # downlink radiated power for each Access Point, in W
    # ATP = 0.2  # 200 mW mobile transmit power
    BSAntG = 0  # base station antenna gain (dB)
    ATAntG = 0  # access terminal antenna gain (dB)
    # noise power (in dBm)
    NP = -230 + 10*np.log10(1.38*(273.15 + 17)) + 30 + 10*np.log10(B) + 60
    MNF = 9  # mobile noise figure in dB
    # BSNF = 9  # base station noise figure in dB
    pLoss = 0  # building penetration loss in dB
    rho_d = np.power(10,
                     (10*np.log10(APP) + 30 + BSAntG + ATAntG - NP - MNF)/10)

    # wavelength = 299792458/(f*10**6)
    # kappa = 2*pi/wavelength
    # For Cell-Free system: antenna coordinates in a disc with radius R
    # Generate random distances for M service antennas in disc with radius R
    d_sa = R*np.sqrt(np.random.uniform(size=(1, M)))
    # Generate random angles for M service antennas in disc
    theta_sa = 2*np.pi*np.random.uniform(size=(1, M))
    x_sa = d_sa*np.cos(theta_sa)  # x-coordinates for the M service antennas
    y_sa = d_sa*np.sin(theta_sa)  # y-coordinates for the M service antennas

    # Generate user coordinates in the disc with radius R
    # Generate random distances for K users in disc with radius R
    d_m = R*np.sqrt(np.random.uniform(size=(1, K)))
    # Generate random angles for K users in disc
    theta_m = 2*np.pi*np.random.uniform(size=(1, K))
    x_m = d_m*np.cos(theta_m)  # x-coordinates for the K users
    y_m = d_m*np.sin(theta_m)  # y-coordinates for the K users

    # Compute the distance from each of the K terminal to each of the M
    # antennas. "ddd" in the following is a MxK matrix.
    ddd = np.sqrt((np.repeat(x_m, M, axis=0)-np.repeat(x_sa.T, K, axis=1))**2
                  + (np.repeat(y_m, M, axis=0)-np.repeat(y_sa.T, K, axis=1))**2
                  + ((hte-hre)/1000)**2)

    # ITU-R propagation model
    PL = 161.04 - 7.1*np.log10(W) + 7.5*np.log10(h)-(24.37-3.7*(h/hte)**2) *\
        np.log10(hte)+(43.42-3.1*np.log10(hte))*(np.log10(ddd*1000)-3)\
        + 20*np.log10(f/1000)-(3.2*(np.log10(11.75*hre))**2-4.97)

    beta = sfstd*np.random.randn(M, K) - pLoss  # Generate shadow fadings
    beta = np.power(10, ((-PL+beta)/10))  # Linear scale
    G = np.sqrt(2)/2*(np.random.randn(M, K)+np.random.randn(M, K)*1j)
    G = np.sqrt(beta)*G
    return G, rho_d


def create_channel_los(f, M, K):
    '''
    Create channel in line-of-sight (LoS) propagation model

    Parameters
    ----------
    f :     frequency in Hz
    M :     number of APs
    K :     number of UEs

    Returns
    -------
    G :     complex channel coefficients matrix
    rho_d : downlink SNR (the power coefficients are normalized by rho_d to
            get a value between 0 and 1).

    '''

    BSP = 0.2  # base station radiated power in watts
    # ATP = 0.2  # mobile station radiated power in watts
    BW = 20  # bandwidth (MHz)
    BSAntG = 0  # base station antenna gain (dB)
    ATAntG = 0  # access terminal antenna gain (dB)
    NP = -230+10*log10(1.38*(273.15+17))+30 + 10 * \
        log10(BW)+60  # noise power in dBm
    MNF = 9  # mobile noise figure in dB
    # BSNF = 9  # base station noise figure in dB
    rho_d = 10**((10*log10(BSP)+30+BSAntG+ATAntG - NP - MNF)/10)
    h_bs = 10  # service antenna height in meters
    h_ms = 1.5  # mobile station antenna height in meters
    R = 500  # radius in m
    wavelength = 299792458/f  # wavelength in meters
    PL_dB = 32.45+20*log10(f*10**(-9))  # free space path loss
    beta = 10**(-PL_dB/10)  # linear scale
    kappa = 2*pi/wavelength

    # generate random distances for M antennas in the cell
    d_a = R*np.sqrt(np.random.uniform(0, 1, (M, 1)))
    # generate random angles for M antennas
    theta_a = 2*pi*np.random.uniform(0, 1, (M, 1))
    xcor_a = d_a*np.cos(theta_a)  # x-coordinates for the M antennas
    ycor_a = d_a*np.sin(theta_a)  # y-coordinates for the M antennas
    zcor_a = h_bs*np.ones((M, 1))  # z-coordinates for the M antennas

    # generate random distances for K users in the cell
    d_u = R*np.sqrt(np.random.uniform(0, 1, (K, 1)))
    # generate random angles for K users
    theta_u = 2*pi*np.random.uniform(0, 1, (K, 1))
    xcor_u = d_u*np.cos(theta_u)  # x-coordinates for the K users
    ycor_u = d_u*np.sin(theta_u)  # y-coordinates for the K users
    zcor_u = h_ms*np.ones((K, 1))  # z-coordinates for the K users
    # DMK = M x K matrix of distances from M antennas to K users
    DMK = np.sqrt(
        (np.repeat(xcor_u, M, axis=1).T-np.repeat(xcor_a, K, axis=1))**2
        + (np.repeat(ycor_u, M, axis=1).T - np.repeat(ycor_a, K, axis=1))**2 +
        (np.repeat(zcor_u, M, axis=1).T-np.repeat(zcor_a, K, axis=1))**2)
    # adjust by free space path loss beta
    G = sqrt(beta)*np.exp(1j*kappa*DMK)/DMK

    return G, rho_d
