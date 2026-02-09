import numpy as np

def weighted_combine(value1, error1, value2, error2):
    """
    Combine two measurements with associated 1σ uncertainties 
    using inverse-variance weighting.

    Parameters
    ----------
    value1, value2 : float
        Measured values.
    error1, error2 : float
        1σ uncertainties on the measurements.

    Returns
    -------
    combined_value : float
        Weighted mean of the two measurements.
    combined_error : float
        Propagated uncertainty of the weighted mean.
    """
    w1 = 1 / error1**2
    w2 = 1 / error2**2
    combined_value = (value1 * w1 + value2 * w2) / (w1 + w2)
    combined_error = np.sqrt(1 / (w1 + w2))
    return combined_value, combined_error



datafile = np.genfromtxt('./viper_outputs/both_nods.vels')
datafile = np.genfromtxt('./11plusone.vels')
datafile = np.genfromtxt('/scratch/home/khoy/Programming/viper/000triple_ddt.vels')
bjds = datafile[:,0]
rvs = datafile[:,1]
errs = datafile[:,2]


mean_bjd, mean_rv, mean_err = [], [], []
for it in range(0,len(bjds),2):
    mean_bjd.append(np.nanmean([bjds[it], bjds[it+1]]))
    tmprv, tmperr = weighted_combine(rvs[it], errs[it], rvs[it+1], errs[it+1])
    mean_rv.append(tmprv)
    mean_err.append(tmperr)

# Convert lists to a 2D array (columns)
output = np.column_stack((mean_bjd, mean_rv, mean_err))

# Save to a space-delimited text file with a header
np.savetxt(
    "comb_nodding_CD35b_plustwo.vels",
    output,
    fmt="%.8f %.8f %.8f",
    comments=""
)