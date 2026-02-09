import numpy as np
import pandas as pd
import pickle
import time
import os
import emcee
import logging
import reddemcee
import kepler
nan = np.nan
gaussian_mixture_objects = dict()
A_ = []
mod_fixed_ = [None, None, None, None, None, None, None, None, None, None, None, None]
cornums = None

logging.getLogger('emcee').setLevel('CRITICAL')

# BEGIN WRITE_DATA_RV FROM MODEL
my_data = pd.read_csv('datalogs/CD352722/run_5/k2/temp/temp_data.csv', index_col=0)

X_ = my_data['BJD'].values
Y_ = my_data['RV'].values
YERR_ = my_data['eRV'].values
ndat = len(X_)


mask1 = (my_data['Flag'] == 1).values
ndat1 = np.sum(mask1)


def my_model(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])

    model0 = np.zeros(ndat)
    err20 = YERR_ ** 2
    per, A, phase, S, C = theta[slice(0, 5, None)]

    ecc = S ** 2 + C ** 2
    if ecc < 1e-6:
        w = 0
    else:
        w = np.arccos(C / (ecc ** 0.5))  # longitude of periastron
        if S < 0:
            w = 2 * np.pi - w

    freq = 2. * np.pi / per
    M = freq * X_ + phase
    E = kepler.solve(M, np.repeat(ecc, len(M)))
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)
    model0 += A * (np.cos(f + w) + ecc * np.cos(w))
    per, A, phase, S, C = theta[slice(5, 10, None)]

    ecc = S ** 2 + C ** 2
    if ecc < 1e-6:
        w = 0
    else:
        w = np.arccos(C / (ecc ** 0.5))  # longitude of periastron
        if S < 0:
            w = 2 * np.pi - w

    freq = 2. * np.pi / per
    M = freq * X_ + phase
    E = kepler.solve(M, np.repeat(ecc, len(M)))
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)
    model0 += A * (np.cos(f + w) + ecc * np.cos(w))
    # Instrumental Part, ins00.model

    model0 += theta[slice(10, 11, None)][1-1] * mask1

    # Instrumental Part, jitter00.model
    
    err20 += mask1 * theta[slice(11, 12, None)][1-1] ** 2

    # End jitter00.model
    return model0, err20


likelihood_constant = - 0.5 * np.log(2*np.pi) * ndat

def my_likelihood(theta):
    model, err2 = my_model(theta)
    return -0.5 * (np.sum((Y_ - model) ** 2 / err2 + np.log(err2))) + likelihood_constant
    

def Hill(x, limits, args):
    kplanets = args[0]
    starmass = args[1]
    periods, amps, eccs = x

    gamma = np.sqrt(1 - eccs)
    sma, minmass = cps(periods, amps, eccs, starmass)
    orden = np.argsort(sma)
    sma = sma[orden]  # in AU
    minmass = minmass[orden]  # in Earth Masses

    periods, amps, eccs = periods[orden], amps[orden], eccs[orden]
    M = starmass * 1047.56 + np.sum(minmass)  # jupiter masses
    mu = minmass / M

    for k in range(kplanets-1):
        alpha = mu[k] + mu[k+1]
        delta = np.sqrt(sma[k+1] / sma[k])

        LHS = alpha**-3 * (mu[k] + (mu[k+1] / (delta**2))) * (mu[k] * gamma[k] + mu[k+1] * gamma[k+1] * delta)**2
        RHS = 1 + (3./alpha)**(4./3) * (mu[k] * mu[k+1])
        if LHS > RHS:
            pass
        else:
            return -np.inf
    return 0.

def GaussianMixture(x, limits, args):
        if limits[0] <= x <= limits[1]:
            return gaussian_mixture_objects[args[0]].score_samples([[x]])
        else:
            return -np.inf

def Isotropic(x, limits, args):
    logZ = args
    if limits[0] <= x <= limits[1]:
        return np.log(0.5*np.sin(x)) - logZ
    return -np.inf

def Jeffreys(x, limits, args):
    low, high = limits
    #ln_norm = np.log(np.log(high/low))
    if low <= x <= high:
        return np.log(1 / (high - low))
    return -np.inf


def Normal(x, limits, args):
    low, high = limits
    mu, s, logZ = args[0], args[1], args[2]
    if low <= x <= high:
        #a, b = (low - mu)/s, (high - mu)/s
        #logZ = np.log(norm.cdf(b) - norm.cdf(a))   # normalising constant
        #return - 0.5 * ((x - mu)/s)**2 - np.log(s*np.sqrt(2*np.pi))
        return -0.5*((x - mu)/s)**2 - np.log(s*np.sqrt(2*np.pi)) - logZ
    return -np.inf

def Beta(x, limits, args):
    if limits[0] <= x <= limits[1]:
        a, b = args[0], args[1]
        return np.log(betapdf.pdf(x, a, b))
    else:
        return -np.inf

def Uniform(x, limits, args):
    logZ = args
    if limits[0] <= x <= limits[1]:
        return logZ
    return -np.inf

def Fixed(x, limits, args):
    return 0.



def my_prior(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])
    lp = 0.

    lp += Uniform(theta[0], [150, 500], -5.857933154483459)

    lp += Uniform(theta[1], [0.001, 500], -6.214606098420192)

    lp += Uniform(theta[2], [0, 6.283185307179586], -1.8378770664093453)

    lp += Normal(theta[3], [-1, 1], [0, np.float64(0.1414213562373095), np.float64(-1.5374368445020986e-12)])

    lp += Normal(theta[4], [-1, 1], [0, np.float64(0.1414213562373095), np.float64(-1.5374368445020986e-12)])

    if lp == -np.inf:
        return lp


    x = theta[slice(0, 5, None)][3]**2 + theta[slice(0, 5, None)][4]**2

    lp += Uniform(x, [0, 1], 0.0)

    lp += Uniform(theta[5], [1, 140], -4.9344739331306915)

    lp += Uniform(theta[6], [0.001, 500], -6.214606098420192)

    lp += Uniform(theta[7], [0, 6.283185307179586], -1.8378770664093453)

    lp += Normal(theta[8], [-1, 1], [0, np.float64(0.1414213562373095), np.float64(-1.5374368445020986e-12)])

    lp += Normal(theta[9], [-1, 1], [0, np.float64(0.1414213562373095), np.float64(-1.5374368445020986e-12)])

    if lp == -np.inf:
        return lp


    x = theta[slice(5, 10, None)][3]**2 + theta[slice(5, 10, None)][4]**2

    lp += Uniform(x, [0, 1], 0.0)

    lp += Uniform(theta[10], [-150, 30], -5.19295685089021)

    if lp == -np.inf:
        return lp


    lp += Normal(theta[11], [0, 60], [0, 10, np.float64(-0.6931471825331207)])

    if lp == -np.inf:
        return lp



    return lp


mypool = None


ntemps, nwalkers, nsweeps, nsteps = 10, 128, 50000, 1
setup = ntemps, nwalkers, nsweeps, nsteps
ndim = 12
betas = [np.float64(1.0), np.float64(0.6392990313331386), np.float64(0.4196001410034899), np.float64(0.26301496597573076), np.float64(0.15328512246705472), np.float64(0.08134161417932831), np.float64(0.040312143183528534), np.float64(0.021454437112932993), np.float64(0.009849746251544687), np.float64(0.002111)]
progress = True


my_backend = None

def set_init():
    pos = np.zeros((ntemps, nwalkers, ndim))

    for t in range(ntemps):
        j = 0

        m = (170.22233964 + 169.13077291) / 2
        r = (170.22233964 - 169.13077291) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (285.02672554 + 270.02177728) / 2
        r = (285.02672554 - 270.02177728) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (4.65168652 + 4.50107606) / 2
        r = (4.65168652 - 4.50107606) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (-0.17961668 + -0.24740771) / 2
        r = (-0.17961668 - -0.24740771) / 2 * 0.707
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (-0.44425418 + -0.50160126) / 2
        r = (-0.44425418 - -0.50160126) / 2 * 0.707
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (140 + 1) / 2
        r = (140 - 1) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (60 + 55) / 2
        r = (60 - 55) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (6.283185307179586 + 0) / 2
        r = (6.283185307179586 - 0) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (1 + -1) / 2
        r = (1 - -1) / 2 * 0.707
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (1 + -1) / 2
        r = (1 - -1) / 2 * 0.707
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (-62.0 + -62.1) / 2
        r = (-62.0 - -62.1) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


        m = (5 + 0) / 2
        r = (5 - 0) / 2 * 1
        dist = np.sort(np.random.uniform(0, 1, nwalkers))
        pos[t][:, j] = r * (2 * dist - 1) + m
        np.random.shuffle(pos[t, :, j])
        j += 1


    return pos



def test_init(max_repeats=100):
    p0 = set_init()

    is_bad_position = True
    repeat_number = 0

    while is_bad_position and (repeat_number < max_repeats):
        is_bad_position = False
        for t in range(ntemps):
            for n in range(nwalkers):
                position_evaluated = p0[t][n]
                if my_prior(position_evaluated) == -np.inf:
                    is_bad_position = True
                    p0[t][n] = set_init()[t][n]
        repeat_number += 1
        if False:
            print('test_init ', repeat_number)

    if is_bad_position:
        print('COULDNT FIND VALID INITIAL POSITION')
    return p0

    
p1 = test_init()

sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             my_likelihood,
                             my_prior,
                             ntemps=ntemps,
                             pool=mypool,
                             backend=my_backend,
                             betas=betas,
                             tsw_history=True,
                             smd_history=True,
                             adapt_tau=100,
                             adapt_nu=1.5,
                             adapt_mode=0
                             )


sampler.D_ = np.array([350.        , 499.999     ,   6.28318531,   2.        ,   2.        ,
 139.        , 499.999     ,   6.28318531,   2.        ,   2.        ,
 180.        ,  60.        ])

def run_thing():
    start = time.time()
    sampler.run_mcmc(p1, nsweeps=nsweeps, nsteps=nsteps, progress=progress)
    end = time.time()

    tts = end - start
    print('temp_script.py took '+str(np.round(tts, 3))+' seconds')
    

    from reddemcee.hdf import PTHDFBackend

    saver = PTHDFBackend('CD352722_run_5.h5')
    saver.reset(ntemps, nwalkers, ndim,
                tsw_hist=True,
                smd_hist=True)

    ntot = sampler.backend.iteration
    saver.grow(ntot)
    with saver.open("a") as f:
        g = f[saver.name]
        g.attrs["iteration"] = ntot


        if sampler.backend.tsw_history_bool:
            g["tsw_history"].resize(ntot, axis=0)
            g["tsw_history"][:] = sampler.backend.tsw_history

        if sampler.backend.smd_history_bool:
            g["smd_history"].resize(ntot, axis=0)
            g["smd_history"][:] = sampler.backend.smd_history


    ntot = sampler.backend[0].iteration
    for t in range(ntemps):
        saver[t].grow(ntot, None)

        with saver[t].open("a") as f:
            g = f[saver[t].name]
            g.attrs["iteration"] = ntot

            g["chain"][:, :, :] = sampler.backend[t].get_chain()
            g["log_like"][:, :] = sampler.backend[t].get_log_like()
            g["log_prob"][:, :] = sampler.backend[t].get_log_prob()
            g["beta_history"][:] = sampler.backend[t].get_betas()
            g["accepted"][:] = sampler.backend[t].accepted

if __name__ == '__main__':
    run_thing()
