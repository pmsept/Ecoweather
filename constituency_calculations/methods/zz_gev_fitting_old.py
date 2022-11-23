import numpy as np
import xarray as xr

from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

importr("evd")

import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Univariate functions calling R

def r_fgev(x):
    rx_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(x))                           # convert X to r-readable vector  
    gev_fit = r['fgev'](rx_r, std_err = "F")                                       # fit GEV using R function
    gev_pars = list(dict(zip(gev_fit.names, list(gev_fit)))['estimate'])           # extract parameter estimates (NB. this is NOT the same parameterisation that would be returned by gev.fit!)
    return np.array(gev_pars)                                                      # output has to be a numpy array to pass to apply_ufunc

def r_qgev(pars, p):

    pars_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(pars))                       # convert parameter vector to r-readable vector
    # if necessary, convert list of percentiles to r-readable vector
    try:
        float(p)
    except:
        # p not a scalar - convert to np. array
        p_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(p))
    else:
        p_r = float(p)
    
    qq = r["qgev"](p = p_r, loc = pars_r[0], scale = pars_r[1], shape = pars_r[2])  # compute quantile(s) according to provided parameters
    return np.array(qq)                                                             # output has to be a numpy array to pass to apply_ufunc

def r_pgev(pars, q):
    
    pars_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(pars))                       # convert parameter vector to r-readable vector
    
    # if necessary, convert list of percentiles to r-readable vector
    try:
        float(q)
    except:
        # q not a scalar - convert to np. array
        q_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(q))
    else:
        q_r = float(q)
    
    pp = r["pgev"](q = q_r, loc = pars_r[0], scale = pars_r[1], shape = pars_r[2])  # compute quantile(s) according to provided parameters
    return np.array(pp)                                                             # output has to be a numpy array to pass to apply_ufunc


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Array functions

def gev_fit(dat):
    u_fit = xr.apply_ufunc(r_fgev, dat, exclude_dims = set(("time",)), input_core_dims=[["time"]], output_core_dims=[["parameter"]], vectorize = True).rename("gev_pars")
    u_fit = u_fit.assign_coords(parameter = ["loc", "scale", "shape"])
    return u_fit

def return_level(pars, y):
    u_rl = xr.apply_ufunc(r_qgev, pars, 1-(1/y), exclude_dims = set(("parameter",)), input_core_dims=[["parameter"], []], vectorize = True).rename("return_levels")
    
    u_rl = u_rl.assign_coords(return_level = "rl" + str(y))
    return u_rl

def return_period(pars, l):
    u_pp = xr.apply_ufunc(r_pgev, pars, l, exclude_dims = set(("parameter",)), input_core_dims=[["parameter"], []], vectorize = True).rename("return_levels")
    rp = 1/(1-u_pp)
    return rp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute per time slice
def fgev_per_slice(da, slice_midpoints = [1996, 2005, 2015, 2025, 2035, 2045, 2055, 2065], slice_length = 30):
    
    # Iterate over years, append DataArray of parameters for each slice
    all_pars = []
    for y in slice_midpoints:
        
        # define time slice
        sl_start = str(y-(int(slice_length/2))-1) + "1201"
        sl_end = str(y+(int(slice_length/2))-1) + "1130"
        dat = da.sel(time = slice(sl_start, sl_end))
        
        # identify months to loop over (single-year time slice)
        dates_out = da.sel(time = slice(str(y-1) + "1201", str(y) + "1130")).time
        
        # fit GEV distribution for each period, append results into a list
        this_year_pars = []
        for m in np.unique(dates_out.dt.month):
            dat_m = dat.sel(time = (dat.time.dt.month == m))
            pars_m = gev_fit(dat_m)
            this_year_pars.append(pars_m)
        
        # concatenate list along time axis & populate time axis
        this_year_pars = xr.concat(this_year_pars, dim = "time")
        this_year_pars["time"] = dates_out
        all_pars.append(this_year_pars)
    
    # concatenate list of DataArrays along time axis & make 'parameters' a coordinate
    all_pars = xr.concat(all_pars, dim = "time")
    all_pars = all_pars.assign_coords(parameter = np.array(["loc", "scale", "shape"]))
    return all_pars


def return_levels(params, years):
    
    if type(years) == int:
        rlevel = xr.apply_ufunc(r_qgev, params, 1-(1/years), exclude_dims = set(("parameter",)), input_core_dims=[["parameter"], []], vectorize = True).rename("rl" + str(years))
    else:
        rlevel = []
        for y in years: 
            u_rl = xr.apply_ufunc(r_qgev, params, 1-(1/y), exclude_dims = set(("parameter",)), input_core_dims=[["parameter"], []], vectorize = True)
            rlevel.append(u_rl)
        rlevel = xr.concat(rlevel, dim = "return_level").assign_coords(return_level = ["rl" + str(y) for y in years])
    rlevel = rlevel.rename("return_levels")
    return rlevel


def return_periods(params, levels):
       
    if type(levels) == float:
        rp = xr.apply_ufunc(r_pgev, params, levels, exclude_dims = set(("parameter",)), input_core_dims=[["parameter"], []], vectorize = True).rename("rp"+str(round(levels)))
    else:
        rp = xr.apply_ufunc(r_pgev, params, levels, exclude_dims = set(("parameter",)), input_core_dims=[["parameter"], []], vectorize = True).rename("return_periods")
        rp = rp.rename(return_level = "return_period")#.assign_coords(return_period = levels.return_level.str.replace("rl","rp"))
    return 1/(1-rp)



def rp_stemplot(da, alias = None, ylim = None):
    
    # Method to produce stem plot of changed return periods with baseline
       
    rp = int(re.sub("rl","",str(da.return_period.values)))
    
    if alias is None:
        plt.stem(da, bottom = rp)
        plt.xlabel("Hydrometric region")
        plt.ylabel("Return period (years)")
        plt.title("Return period of current once-in-"+str(rp)+"-years event")
        if ylim is not None: plt.ylim(ylim[0],ylim[1])
    else:
        alias.stem(da, bottom = rp)
        alias.set_xlabel("Hydrometric region")
        alias.set_ylabel("Return period (years)")
        alias.set_title("Return period of current once-in-"+str(rp)+"-years event")
        if ylim is not None: alias.set_ylim(ylim[0],ylim[1])