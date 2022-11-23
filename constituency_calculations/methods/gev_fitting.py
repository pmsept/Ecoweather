
import xarray as xr
import numpy as np
import xclim
import re

from xclim.indices.stats import frequency_analysis, fit, fa, parametric_quantile
from xclim.indices.generic import select_resample_op
from scipy.stats import genextreme as gev

import warnings
warnings.filterwarnings("ignore", message = ".+value encountered.+")     # warnings to do with default behaviour of method are quite annoying

from IPython.display import clear_output

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fit_gev(da, slices):
    
    # fit GEV parameters to all time slices (seasonal + annual maxima)
    
    sl_params = []
    for sl in slices:
        
        # fix dates if necessary, select period of interest
        if xclim.core.calendar.get_calendar(da) == "360_day": sl = re.sub("31$", "30", sl)
        da_sl = da.sel(time = slice(sl[:8], sl[9:]))
        s_maxima = select_resample_op(da_sl, op="max", freq="QS-DEC")
        
        # fit seasonal parameters
        params = []
        for s in ["DJF", "MAM", "JJA", "SON"]:
            s_params = fit(s_maxima.where(~np.isnan(s_maxima)).sel(time = s_maxima.time.dt.season == s), dist="genextreme").expand_dims(season = [s])
            params.append(s_params)
            print(".", end = "")
        
        # fit annual parameters
        a_maxima = select_resample_op(da_sl, op="max", freq="AS-DEC")
        a_params = fit(a_maxima.where(~np.isnan(a_maxima)), dist="genextreme").expand_dims(season = ["annual"])
        params.append(a_params)
        
        sl_params.append(xr.concat(params, "season"))
        print("-", end = "")
        
    sl_params = xr.concat(sl_params, "period").assign_coords(period = slices)
    
    return sl_params



def returnperiod(pars, l):
    
    # xclim doesn't provide a function to compute return periods so apply over each set of parameters
    # rounded to hundredths of a year
    u_rp = xr.apply_ufunc(lambda pars, l : np.array(1/(1-gev.cdf(l, *pars))), pars, l, exclude_dims = set(("dparams",)), input_core_dims=[["dparams"], []], vectorize = True).round(2)
    return u_rp