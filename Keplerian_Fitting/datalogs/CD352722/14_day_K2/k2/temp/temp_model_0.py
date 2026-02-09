
# BEGIN WRITE_DATA_RV FROM MODEL
my_data = pd.read_csv('datalogs/CD352722/run_3/k2/temp/temp_data.csv', index_col=0)

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


