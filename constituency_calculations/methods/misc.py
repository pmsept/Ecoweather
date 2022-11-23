import numpy as np
import xarray as xr
import pandas as pd
import glob
import subprocess
import re
import xclim

from IPython.display import clear_output

import sys
sys.path.append('/data/met/processing/10_methods')
from dicts import cmip5_13, cmip5_ec
from regridding import map2vec, vec2map

xy = ["projection_x_coordinate", "projection_y_coordinate"]

pd.options.display.max_rows = None


def load_eval(varnm, season, period = "19890101-20081231"):
        
    # load rcm runs, split into evaluation & gcm-driven
    rcms = xr.open_mfdataset("/data/met/ukcordex/*/*/*/clim/"+varnm+"_12km_*_eval-climatology.nc").sel(season = season)[varnm].load()
    evals = rcms.sel(run = [run for run in rcms.run.values if "ECMWF" in run])
    rcms = rcms.sel(run = [run for run in rcms.run.values if not "ECMWF" in run])
    
    # load gcms, split into CMIP5-13 and EuroCORDEX ensembles
    gcms = xr.open_mfdataset("/data/met/cmip5/*/*/clim/"+varnm+"_*_eval-climatology.nc").sel(season = season)[varnm].load()
    gcm_13 = gcms.sel(run = np.isin(gcms.run, [r for r in gcms.run.values if r in cmip5_13]))
    gcm_ec = gcms.sel(run = np.isin(gcms.run, [r for r in gcms.run.values if r in cmip5_ec]))
    
    # load both sets of UKCP18 runs
    ukcp_r = xr.open_mfdataset("/data/met/ukcp18/*/clim/"+varnm+"_rcp85_ukcp18_12km_*_eval-climatology.nc").sel(season = season)[varnm].load()
    if varnm in ["prcprop", "prc", "prf"]:
        obs = xr.open_mfdataset("/data/met/hadUK-grid/clim/pr_hadukgrid_uk_12km_*_eval-climatology.nc").sel(season = season).pr.load()
        obs = xr.zeros_like(obs)
        return {"obs" : obs, "CMIP5-13" : gcm_13, "CMIP5-EC" : gcm_ec, "EuroCORDEX" : rcms, "ERA-EuroCORDEX" : evals, "UKCP18 12km" : ukcp_r}
    else:
        obs = xr.open_mfdataset("/data/met/hadUK-grid/clim/"+varnm+"_hadukgrid_uk_12km_*_eval-climatology.nc").sel(season = season)[varnm].load()
        ukcp_g = xr.open_mfdataset("/data/met/ukcp18/60km/*/clim/"+varnm+"_rcp85_*_eval-climatology.nc").sel(season = season)[varnm].load()
        return {"obs" : obs, "CMIP5-13" : gcm_13, "CMIP5-EC" : gcm_ec, "EuroCORDEX" : rcms, "ERA-EuroCORDEX" : evals, "UKCP18 60km" : ukcp_g, "UKCP18 12km" : ukcp_r}


def load_diff(varnm, season = None, period = None, rel_diff = False, recover_clim = False):
       
    # load all the data
    rcms = xr.open_mfdataset("/data/met/ukcordex/*/*/*/anom/"+varnm+"_*_seasonal-anomaly.nc")[varnm].load()
    gcms = xr.open_mfdataset("/data/met/cmip5/*/*/anom/"+varnm+"_*_seasonal-anomaly.nc")[varnm].load()
    ukcp_r = xr.open_mfdataset("/data/met/ukcp18/*/anom/"+varnm+"_*_seasonal-anomaly.nc")[varnm].load()
    
    # prc not available for 60km runs
    if not varnm in ["prcprop", "prc", "prf"]:
        ukcp_g = xr.open_mfdataset("/data/met/ukcp18/60km/*/anom/"+varnm+"_*_seasonal-anomaly.nc")[varnm].load()
    else:
        ukcp_g = xr.ones_like(ukcp_r) * np.nan
    
    if rel_diff: 
        # compute relative difference wrt baseline period
        rcms, gcms, ukcp_r, ukcp_g = [da.isel(period = slice(1,8)) / da.isel(period = 0) * 100 for da in [rcms, gcms, ukcp_r, ukcp_g]]
       
    if recover_clim: 
        # add baseline period to anomaly to recover original climatology
        rcms, gcms, ukcp_r, ukcp_g = [xr.concat([da.isel(period = 0), da.isel(period = slice(1,8)) + da.isel(period = 0)], "period") for da in [rcms, gcms, ukcp_r, ukcp_g]]
    
    if not (not season): 
        rcms, gcms, ukcp_r, ukcp_g = [da.sel(season = season) for da in [rcms, gcms, ukcp_r, ukcp_g]]
    
    if not (not period):
        rcms, gcms, ukcp_r, ukcp_g = [da.sel(period = period) for da in [rcms, gcms, ukcp_r, ukcp_g]]
            
    gcm_13 = gcms.sel(run = np.isin(gcms.run, [r for r in gcms.run.values if r in cmip5_13]))
    gcm_ec = gcms.sel(run = np.isin(gcms.run, [r for r in gcms.run.values if r in cmip5_ec]))
    
    if varnm in ["prcprop", "prc", "prf"]:
        return {"CMIP5-13" : gcm_13, "CMIP5-EC" : gcm_ec, "EuroCORDEX" : rcms, "UKCP18 12km" : ukcp_r}
    else:
        return {"CMIP5-13" : gcm_13, "CMIP5-EC" : gcm_ec, "EuroCORDEX" : rcms, "UKCP18 60km" : ukcp_g, "UKCP18 12km" : ukcp_r}



def tabulate(ftype = "anom/*.nc", details = False, filterby = None):
    
    incl = []
    
    # Method to quickly tabulate all available runs
    if len(glob.glob("/data/met/ukcordex/*/*/*/"+ftype)) > 0:
        ukcordex = pd.DataFrame([x.split("/") for x in glob.glob("/data/met/ukcordex/*/*/*/"+ftype)])[[3,4,5,6,8]]
        ukcordex["run"] = ukcordex[4]+"_"+ukcordex[6]+"_"+ukcordex[5]
        ukcordex["varnm"] = ukcordex[8].str.replace("_.+","",regex = True)
        ukcordex[3] = "EuroCORDEX"
        incl.append(ukcordex)
    
    if len(glob.glob("/data/met/ukcp18/[0-9][0-9]/"+ftype)) > 0:
        ukcp_12k = pd.DataFrame([x.split("/") for x in glob.glob("/data/met/ukcp18/[0-9][0-9]/"+ftype)])[[3,4,6]]
        ukcp_12k["run"] = "ukcp_12km"+"_"+ukcp_12k[4]
        ukcp_12k["varnm"] = ukcp_12k[6].str.replace("_.+","",regex = True)
        ukcp_12k[3] = "UKCP 12k"
        incl.append(ukcp_12k)
    
    if len(glob.glob("/data/met/ukcp18/60km/[0-9][0-9]/"+ftype)) > 0:
        ukcp_60k = pd.DataFrame([x.split("/") for x in glob.glob("/data/met/ukcp18/60km/[0-9][0-9]/"+ftype)])[[3,5,7]]
        ukcp_60k["run"] = "ukcp_60km"+"_"+ukcp_60k[5]
        ukcp_60k["varnm"] = ukcp_60k[7].str.replace("_.+","",regex = True)
        ukcp_60k[3] = "UKCP 60k"
        incl.append(ukcp_60k)
    
    if len(glob.glob("/data/met/cmip5/*/*/"+ftype)) > 0:
        cmip5 = pd.DataFrame([x.split("/") for x in glob.glob("/data/met/cmip5/*/*/"+ftype)])[[3,4,5,7]]
        cmip5["run"] = cmip5[4]+"_"+cmip5[5]
        cmip5["varnm"] = cmip5[7].str.replace("_.+","",regex = True)
        cmip5[3] = "CMIP5"
        incl.append(cmip5)
        
    df = pd.concat([x[[3, "run", "varnm"]] for x in incl]).rename(columns = {3:"src"})
    
    if filterby:
        df = df[eval(filterby)]
    
    if details:
        return pd.crosstab(index=[df.src, df.run], columns=df.varnm)
    else:
        return pd.crosstab(index=df.src, columns=df.varnm)
    
    
def run_name(fpath):
    
    # method to extract model name from path string
    fp_split = fpath.split("/")
    
    if "ukcordex" in fpath: 
        run_name = fp_split[4]+"_"+fp_split[6]+"_"+fp_split[5]
    elif "60km" in fpath:   
        run_name = fp_split[5]
    elif "cmip5" in fpath:  
        run_name = fp_split[4]+"_"+fp_split[5]
    else:
        run_name = fp_split[4]
        
    return run_name


def all_files(pattern):
    
    return glob.glob("/data/met/ukcp18/60km/*/"+pattern) + glob.glob("/data/met/cmip5/*/*/"+pattern) + glob.glob("/data/met/ukcordex/*/*/r[0-9]*/"+pattern) + glob.glob("/data/met/ukcp18/[0-9][0-9]/"+pattern)



def aggregate_slices(da, aggregate_by = "mean", slice_length = 30):
    
    # Method to aggregate a DataArray into 20- or 30-year time slices
    
    if slice_length == 30:
        slice_midpoints = [1996, 2005, 2015, 2025, 2035, 2045, 2055, 2065]
    elif slice_length == 20:
        slice_midpoints = [1991, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070]
    else:
        print("Slice length must be either 20 or 30")
    
    all_slices = []
    for y in slice_midpoints:
        
        # define time slice
        sl_start = str(y-(int(slice_length/2))-1) + "1201"
        sl_end = str(y+(int(slice_length/2))-1) + "1130"
        dat = da.sel(time = slice(sl_start, sl_end))
        
        # identify months to loop over (single-year time slice)
        dates_out = da.sel(time = slice(str(y-1) + "1201", str(y) + "1130")).time
        
        # aggregate for each period
        current_slice = []
        for d in np.unique(dates_out.dt.dayofyear):
            dat_d = dat[dat.time.dt.dayofyear == d,:,:]
            sl = getattr(dat_d, aggregate_by)(dim="time")
            current_slice.append(sl)
        
        # concatenate list along time axis & populate time axis
        current_slice = xr.concat(current_slice, dim = "time")
        current_slice["time"] = dates_out
        all_slices.append(current_slice)
        
    # concatenate list of DataArrays along time axis
    all_slices = xr.concat(all_slices, dim = "time")
    return all_slices


def osgb_region(da):
    
    # select a subset of data lat-lon coordinates
    if "longitude" in da.coords:
        subset = da.sel(longitude = slice(-13.1, 4.6), latitude = slice(48.6, 60.9))
    elif "lon" in da.coords:
        subset = da.sel(lon = slice(-13.1, 4.6), lad = slice(48.6, 60.9))
    else:
        print("Can't find longitude coordinates")
        subset = da
    
    return subset



def epdf(da, minlength = 0, binwidth = 1):
    
    # method to compute empirical PDF for DAV metric from DataArray
    # currently tailored to precipitation (because thresholded at 1mm/day)
       
    # flatten all values into a vector, remove missing values 
    
    # convert to integers by truncating so that all values between 1 and 2 are labelled 1, values between 2 and 3 are labelled 2, and so on.
    vals = [int(x // binwidth) for x in da]
    
    # count occurrences of each value & compute midpoint of each bin
    xcounts = np.bincount(vals, minlength = minlength)
    xvalues = [(x + 0.5) * binwidth for x in list(range(max(vals + [minlength])+1))]
    
    # get total precipitation to normalise the distribution
    tval = da.sum()
    
    # weight frequencies by midpoints, normalise by dividing by total
    xpdf = [n * x for n, x in zip(xcounts, xvalues)] / tval
    
    return xpdf



def get_ndays(fnm):
    
    # Method to strip out time dimension from ncdump. 
    # Very hacky but works so far...
    p1 = subprocess.Popen(["ncdump", "-h", fnm], stdout = subprocess.PIPE)
    
    ncdump_out = bytes.decode(p1.communicate()[0])
    timestring = [x for x in re.sub("UNLIMITED ;", "UNLIMITED",ncdump_out).split(";") if "\ttime = " in x][0]
    
    if "UNLIMITED" in timestring:
        nt = int(re.findall("[0-9]+", re.findall("\([0-9]+.+\)", timestring)[0])[0])
    else:
        nt = int(re.sub("time = ", "", re.findall("time = [0-9]{5}", timestring)[0]))
    
    return nt



