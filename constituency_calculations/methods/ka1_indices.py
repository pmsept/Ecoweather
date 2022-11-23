import xarray as xr
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def r99ptot_seasonal(da):  
    
    # keeping this function because used elsewhere, but now a wrapper only
    return rXXptot_seasonal(da, 99)


def rXXptot(da, q):
    
    if q < 1: q = q * 100
        
    contrib = xr.concat([rXXptot_seasonal(da, q), rXXptot_annual(da, q).expand_dims(season = ["annual"])], "season")
    contrib = contrib.rename("r"+str(q)+"ptot").reset_coords("quantile", drop = True)
    
    return contrib

def rXXptot_seasonal(da, q):
    
    if q < 1: q = q * 100
    # Method to compute fraction of precip exceeding a percentile threshold per season
    # could easily be modified to compute over any time period
    # have also hard-coded threshold to only include wet-day precipitation
    
    # percentiles of wet-day precipitation per season
    rXX = da.where(da >= 1).groupby("time.season").quantile(q / 100)
    
    # resample individual seasons to full time series (based on xclim.core.calendar.resample_doy)
    rXX_ts = rXX.rename(season="time").reindex(time = da.time.dt.season).assign_coords(time = da.time)
    
    # compute total contribution from extreme rainfall & total precip
    rXX_contrib = da.where(da >= rXX_ts).groupby("time.season").sum()
    pr_total = da.where(da >= 1).groupby("time.season").sum()
    
    rXXptot = (rXX_contrib / pr_total).rename("r"+str(q)+"ptot") * 100
    
    return rXXptot


def rXXptot_annual(da, q):
    
    if q < 1: q = q * 100
    # Method to compute fraction of precip exceeding a percentile threshold per season
    # could easily be modified to compute over any time period
    # have also hard-coded threshold to only include wet-day precipitation
    
    # percentiles of wet-day precipitation per season
    rXX = da.where(da >= 1).quantile(q / 100, dim = "time")
        
    # compute total contribution from extreme rainfall & total precip
    rXX_contrib = da.where(da >= rXX).sum(dim = "time")
    pr_total = da.where(da >= 1).sum(dim = "time")
    
    rXXptot = (rXX_contrib / pr_total).rename("r"+str(q)+"ptot") * 100
    
    return rXXptot


def r99ptot_prctot(pr, prc):
    
    # Method to compute fraction of precip exceeding a percentile threshold per season arising from convective processes
    # could easily be modified to compute over any time period
    # have also hard-coded threshold to only include wet-day precipitation
    
    # percentiles of wet-day precipitation per season
    r99 = pr.where(pr >= 1).groupby("time.season").quantile(0.99)
    
    # resample individual seasons to full time series (based on xclim.core.calendar.resample_doy)
    r99_ts = r99.rename(season="time").reindex(time=pr.time.dt.season).assign_coords(time = pr.time)
    
    # compute total contribution from extreme rainfall & total precip
    r99ptot_pr = pr.where(pr > r99_ts).groupby("time.season").sum()
    r99ptot_prc = prc.where(pr > r99_ts).groupby("time.season").sum()
    
    prc99ptot = (r99ptot_prc / r99ptot_pr).rename("prc99ptot") * 100
    
    return prc99ptot




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

importr("SPEI")

def r_spi(pr, nmonths = 6):
    
    # pr is a vector of monthly total precipitation
    
    if sum(np.isnan(pr)) == len(pr):
        return pr
    
    # convert x to r-readable vector
    x_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(pr))     
    
    # convert required arguments to r-readable objects
    ts_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1980,12]))
    cal_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1980,12]))      # All values are normalised against the baseline period
    cal_end = rpy2.robjects.numpy2ri.numpy2rpy(np.array([2010,11]))        # All values are normalised against the baseline period
    
    # first, convert x to r time series object. Will assume monthly data for now but final function should allow flexibility
    x_ts = r["ts"](x_r, freq = 12, start = ts_start)
    
    # Then we can fit the SPI series using the default modelling parameters
    spi_out = r["spi"](x_ts, scale = nmonths, ref_start = cal_start, ref_end = cal_end)
    
    # output has to be a numpy array to play nicely with xr.apply_ufunc
    return np.array(list(dict(zip(spi_out.names, list(spi_out)))['fitted']))


def r_spei(epr, nmonths = 6):
    
    # epr is a vector of monthly total effective precipitation (total precip - total pet for the month)
    
    if sum(np.isnan(epr)) == len(epr):
        return epr
    
    # convert x to r-readable vector
    x_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(epr))     
    
    # convert required arguments to r-readable objects
    ts_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1980,12]))
    cal_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1980,12]))      # All values are normalised against the baseline period
    cal_end = rpy2.robjects.numpy2ri.numpy2rpy(np.array([2010,11]))        # All values are normalised against the baseline period
    
    # first, convert x to r time series object. Will assume monthly data for now but final function should allow flexibility
    x_ts = r["ts"](x_r, freq = 12, start = ts_start)
    
    # Then we can fit the SPEI series using the default modelling parameters
    spei_out = r["spei"](x_ts, scale = nmonths, ref_start = cal_start, ref_end = cal_end)
    
    # output has to be a numpy array to play nicely with xr.apply_ufunc
    return np.array(list(dict(zip(spei_out.names, list(spei_out)))['fitted']))



# wrappers for R functions over evaluation period (same as before, but with different time periods: easier than working out how to pass additional parameters to apply_ufunc)

def r_spi_eval(pr, nmonths = 6):
    
    # pr is a vector of monthly total precipitation
    if sum(np.isnan(pr)) == len(pr):
        return pr
    
    # convert x to r-readable vector
    x_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(pr))     
    
    # convert required arguments to r-readable objects
    ts_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1988,1]))
    cal_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1989,1]))      # All values are normalised against the baseline period
    cal_end = rpy2.robjects.numpy2ri.numpy2rpy(np.array([2008,12]))        # All values are normalised against the baseline period
    
    # first, convert x to r time series object. Will assume monthly data for now but final function should allow flexibility
    x_ts = r["ts"](x_r, freq = 12, start = ts_start)
    
    # Then we can fit the SPI series using the default modelling parameters
    spi_out = r["spi"](x_ts, scale = nmonths, ref_start = cal_start, ref_end = cal_end)
    
    # output has to be a numpy array to play nicely with xr.apply_ufunc
    return np.array(list(dict(zip(spi_out.names, list(spi_out)))['fitted']))


def r_spei_eval(epr, nmonths = 6):
    
    # epr is a vector of monthly total effective precipitation (total precip - total pet for the month)
    if sum(np.isnan(epr)) == len(epr):
        return epr
    
    # convert x to r-readable vector
    x_r = rpy2.robjects.numpy2ri.numpy2rpy(np.array(epr))     
    
    # convert required arguments to r-readable objects
    ts_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1988,1]))
    cal_start = rpy2.robjects.numpy2ri.numpy2rpy(np.array([1989,1]))      # All values are normalised against the baseline period
    cal_end = rpy2.robjects.numpy2ri.numpy2rpy(np.array([2008,12]))        # All values are normalised against the baseline period
    
    # first, convert x to r time series object. Will assume monthly data for now but final function should allow flexibility
    x_ts = r["ts"](x_r, freq = 12, start = ts_start)
    
    # Then we can fit the SPEI series using the default modelling parameters
    spei_out = r["spei"](x_ts, scale = nmonths, ref_start = cal_start, ref_end = cal_end)
    
    # output has to be a numpy array to play nicely with xr.apply_ufunc
    return np.array(list(dict(zip(spei_out.names, list(spei_out)))['fitted']))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def nearest_px(arr, x, y):
    
    if "projection_x_coordinate" in arr.coords:
        lats = arr.projection_y_coordinate
        lons = arr.projection_x_coordinate
    elif "latitude" in arr.coords:
        lats = arr.latitude
        lons = arr.longitude
    else:
        lats = arr.lat
        lons = arr.lon
    
    dist2 = np.multiply(lons - x, lons - x) + np.multiply(lats - y, lats - y)
    ind = np.unravel_index(dist2.argmin(axis=None), dist2.shape)
    return ind

    
def nao(psl, freq = "QS-DEC"):
      
    if "grid_latitude" in psl.coords: 
        # UKCP grids don't have lat & lon. Use hard-coded indices instead.
        Gibraltar = [82, 104]
        Iceland = [101, 382]
    else:
        # find closest point in grid of latitudes for all other model output
        Gibraltar = nearest_px(psl, -5.35, 36.14)
        Iceland = nearest_px(psl, -22.7, 65.07)
    
    
    # NAO used by UKCP18 is difference in pressure between those two grid cells
    nao = psl[:,Gibraltar[1],Gibraltar[0]] - psl[:,Iceland[1],Iceland[0]]
    nao = nao.resample(time = freq).mean(dim = "time").rename("nao").assign_attrs(units = psl.units)
    
    return(nao)


