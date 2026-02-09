
# BEGIN WRITE_DATA_RV FROM MODEL
my_data = pd.read_csv('datalogs/CD352722/run_5/k0/temp/temp_datafull_model.csv', index_col=0)

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
    # Instrumental Part, ins00.model

    model0 += theta[slice(0, 1, None)][1-1] * mask1

    # Instrumental Part, jitter00.model
    
    err20 += mask1 * theta[slice(1, 2, None)][1-1] ** 2

    # End jitter00.model
    return model0, err20


