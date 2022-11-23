import xarray as xr
import numpy as np
import pandas as pd
import string
import glob
import re

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['savefig.bbox'] = "tight"    # always save with tight bounding box

import sys
sys.path.append('/data/met/processing/10_methods')
import regridding as rg
from taylorDiagram import TaylorDiagram
from misc import load_eval, load_diff
from dicts import *

import warnings
warnings.filterwarnings("ignore", message = ".+multi-part geometry.+")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# markers etc for different GCMs

gcm_markers = {'HadUK-Grid'                   : '*',
               'ECMWF-ERAINT_r1i1p1'          : '*',
               'CNRM-CERFACS-CNRM-CM5_r1i1p1' : 'o',
               'ICHEC-EC-EARTH_r12i1p1'       : 'p',
               'ICHEC-EC-EARTH_r1i1p1'        : 'h',
               'ICHEC-EC-EARTH_r3i1p1'        : 'H',
               'IPSL-IPSL-CM5A-MR_r1i1p1'     : "s",
               'MOHC-HadGEM2-ES_r1i1p1'       : 'P',
               'MPI-M-MPI-ESM-LR_r1i1p1'      : '<',
               'MPI-M-MPI-ESM-LR_r2i1p1'      : '^',
               'MPI-M-MPI-ESM-LR_r3i1p1'      : '>',
               'NCC-NorESM1-M_r1i1p1'         : 'X',
               'UKCP18 12km'                  : '+',
               'UKCP18 60km'                  : 'x',
               'ACCESS1-3_r1i1p1'             : 'o',
               'BCC-CSM1-1_r1i1p1'            : 'p',            
               'CanESM2_r1i1p1'               : 'h',
               'CCSM4_r1i1p1'                 : 'H',            
               'CESM1-BGC_r1i1p1'             : 's', 
               'CMCC-CM_r1i1p1'               : '>', 
               'GFDL-ESM2G_r1i1p1'            : 'd', 
               'MPI-ESM-MR_r1i1p1'            : '^',
               'MRI-CGCM3_r1i1p1'             : '<',
               'ERAINT_r1i1p1'                : '*',
               'CNRM-CM5_r1i1p1'              : 'o',
               'EC-EARTH_r12i1p1'             : 'p',
               'EC-EARTH_r1i1p1'              : 'h',
               'EC-EARTH_r3i1p1'              : 'H',
               'IPSL-CM5A-MR_r1i1p1'          : "s",
               'HadGEM2-ES_r1i1p1'            : 'P',
               'MPI-ESM-LR_r1i1p1'            : '^',
               'MPI-ESM-LR_r2i1p1'            : '<',
               'MPI-ESM-LR_r3i1p1'            : '>',
               'NorESM1-M_r1i1p1'             : 'X',
               'Other CMIP5'                  : '.'}

rcm_colours = {'ALADIN63'          : 'mediumblue',
               'CCLM4-8-17'        : 'blueviolet',
               'COSMO-crCLIM-v1-1' : 'mediumvioletred',
               'HIRHAM5'           : 'red',
               'HadREM3-GA7-05'    : 'darkorange',
               'RACMO22E'          : 'gold',
               'RCA4'              : 'yellowgreen',
               'REMO2015'          : 'green',
               'RegCM4-6'          : 'darkturquoise',
               'WRF381P'           : 'dodgerblue',
               'UKCP1801'          : 'darkorange',
               'UKCP18'            : 'black'}

def run_markers(da): return [gcm_markers[g] if "_" in g else "$"+g+"$" for g in da.run.str.replace("p1_.+","p1").values]
def run_colours(da): return ["black" if len(r) == 2 else "white" if "_" in r else rcm_colours[r] for r in da.run.str.replace(".+p1_","").values]
    
def gcm_colours(gcm_names = list(gcm_markers.keys())):
    
    cols = ["black" if (gcm_nm in ["HadUK-Grid"]) else "white" if (gcm_nm in cmip5_ec) or (gcm_nm in list(gcm_full_names.values())) or (gcm_nm in ["ECMWF-ERAINT_r1i1p1", "ERAINT_r1i1p1"]) else "grey" for gcm_nm in gcm_names]
    if len(cols) == 1:
        return cols[0]
    else:
        return cols

# extract colours/lines for legend
gcm_handles = [matplotlib.lines.Line2D([], [], color = gcm_colours([gcm_nm]), marker = m, markersize = 6, markeredgecolor = "black", linestyle = "None") for gcm_nm, m in gcm_markers.items()]
rcm_handles = [matplotlib.lines.Line2D([], [], color = c, marker = 'o', markersize = 6, markeredgecolor = "black", linestyle = "None") for rcm_nm, c in rcm_colours.items()]


# season labels
seas = {"DJF" : "winter", "MAM" : "spring", "JJA" : "summer", "SON" : "autumn"}
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gcm_legend(gcm_names = list(gcm_markers.keys())[:11], ax = None, title = "GCM", markersize = 6, short_names = False, **kwargs):
    
    if not ax: ax = plt.gcf()
    # add legend & title
    if short_names:
        gcm_labels = [{v : k for k, v in gcm_full_names.items()}[gcm_nm] for gcm_nm in list(gcm_markers.keys()) if gcm_nm in gcm_names]
    else:
        gcm_labels = [gcm_nm for gcm_nm in list(gcm_markers.keys()) if gcm_nm in gcm_names]
    gcm_handles = [matplotlib.lines.Line2D([], [], color = gcm_colours([gcm_nm]), marker = m, markersize = markersize, markeredgecolor = "black", linestyle = "None") for gcm_nm, m in gcm_markers.items() if gcm_nm in gcm_names]
    
    ax.legend(handles = gcm_handles, labels = gcm_labels, edgecolor = "white", title = title, **kwargs)
    

def rcm_legend(rcm_names = list(rcm_colours.keys())[:10], ax = None, title = "RCM", markersize = 6, **kwargs):
    
    if not ax: ax = plt.gcf()
        
    # add legend & title
    rcm_labels = [rcm_nm for rcm_nm in list(rcm_colours.keys()) if rcm_nm in rcm_names]
    rcm_handles = [matplotlib.lines.Line2D([], [], color = c, marker = 'o', markersize = markersize, markeredgecolor = "black", linestyle = "None") for rcm_nm, c in rcm_colours.items() if rcm_nm in rcm_names]
    
    if "UKCP18" in rcm_names: rcm_handles[rcm_names.index("UKCP18")] = matplotlib.lines.Line2D([], [], color = "black", marker = "+", markersize = markersize, linestyle = "None")
    if "UKCP1801" in rcm_names: rcm_handles[rcm_names.index("UKCP1801")] = matplotlib.lines.Line2D([], [], color = rcm_colours["HadREM3-GA7-05"], marker = "+", markersize = markersize, linestyle = "None")

    ax.legend(handles = rcm_handles, labels = rcm_labels, edgecolor = "white", title = title, **kwargs)
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Methods to get range of DataArray and adjust colourmap/colourbar accordingly

def trim_da(da):
    
    # method to set a sensible limit for plotting
    qq = da.quantile([0.01, 0.25, 0.75, 0.99])
    iqr = (qq.sel(quantile = 0.75) - qq.sel(quantile = 0.25))
    ul = qq.sel(quantile = 0.99) + 5 * iqr
    ll = qq.sel(quantile = 0.01) - 5 * iqr
    
    if qq.sel(quantile = 0.01) < 0:
        da = da.where(da <= ul).where(da >= ll)
    else: 
        da = da.where(da <= ul)
    
    return da



    
def vrange(da):
    
    # Method to get min & max of values for plotting
    if type(da) == xr.core.dataarray.DataArray:
        da = da.reset_coords(drop = True)
        da = da.where(np.isfinite(da))
        vmin = da.min().values
        vmax = da.max().values
    if type(da) == list:
        vmin = min(da)
        vmax = max(da)
    else:
        vmin = da.min()
        vmax = da.max()        

    if (np.sign(vmin) != 0) and (np.sign(vmax) != 0) and (np.sign(vmin) != np.sign(vmax)):
        vmax = np.abs([vmin, vmax]).max()
        vmin = -vmax
    
    return {"vmin" : vmin, "vmax" : vmax}


def frange(da):
    
    # method to set a sensible limit for plotting
    qq = da.quantile([0.01, 0.25, 0.75, 0.99])
    iqr = (qq.sel(quantile = 0.75) - qq.sel(quantile = 0.25))
    ul = qq.sel(quantile = 0.99) + 5 * iqr
    ll = qq.sel(quantile = 0.01) - 5 * iqr
       
    flims = vrange(da.where(da <= ul).where(da >= ll).reset_coords(drop = True))
    
    if da.max() > ul:
        fmax = flims["vmax"]
    else:
        fmax = da.max()
    if da.min() < ll:
        fmin = flims["vmin"]
    else:
        fmin = da.min()
        
    return { "fmin" : fmin, "fmax" : fmax}


def fix_cmap(cmap, vmin, vmax):
    
    # replace diverging colourmap with sequential colourmap where values do not include 0
    
    if not cmap in ["PuOr", "PuOr_r", "RdBu", "RdBu_r", "PRGn", "PRGn_r"]: return cmap
    
    if vmin < 0 and vmax > 0: # diverging colourmap
        return cmap
    
    if vmin >= 0: # sequential, positive
        return {"PuOr_r" : "Purples", "PuOr" : "Oranges", "RdBu_r" : "Reds", "RdBu" : "Blues", "PRGn" : "Greens", "PRGn_r" : "Purples"}[cmap]
    
    if vmax <= 0: # sequential, negative
        return {"PuOr" : "Oranges_r", "PuOr_r" : "Purples_r", "RdBu" : "Reds_r", "RdBu_r" : "Blues_r", "PRGn" : "Purples", "PRGn_r" : "Greens"}[cmap]
    
    
def fix_cbar(fmin, fmax, vmin, vmax):
    
    if fmin is None: fmin = vmin
    if fmax is None: fmax = vmax
    
    # if data extends beyond the limits set, constrain the colourbar
    if fmin > vmin and fmax < vmax:
        return {"vmin" : fmin, "vmax" : fmax, "extend" : "both"}
    elif fmin > vmin:
        return {"vmin" : fmin, "vmax" : vmax, "extend" : "min"}
    elif fmax < vmax:
        return {"vmin" : vmin, "vmax" : fmax, "extend" : "max"}
    else:
        return {"vmin" : vmin, "vmax" : vmax, "extend" : "neither"}
    
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def qscatter(x, y, ax = None, xlab = "First principal component", ylab = "Second principal component", title = "", c = list(rcm_colours.values()), m = None, edgecolors = "black", zorder = 99, *args, **kwargs):
    
    # quick scatterplot: by default, uses RCM colours
    mscatter(x, y, ax = ax, c = c, m = m, edgecolors = edgecolors, zorder = zorder, *args, **kwargs)
    
    if ax is None:
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axvline(0, color = "grey", linestyle = "--")
        plt.axhline(0, color = "grey", linestyle = "--")
        plt.title(title)
    else:
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.axvline(0, color = "grey", linestyle = "--")
        ax.axhline(0, color = "grey", linestyle = "--")
        ax.set_title(title)


        
# hacked scatterplot to allow list of markers (https://github.com/matplotlib/matplotlib/issues/11155)
def mscatter(x,y,ax=None, m=None, **kw):
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, matplotlib.markers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = matplotlib.markers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc



def ukmap_subplots(ncols = 1, nrows = 1, figsize = None, **kwargs):
    
    if figsize is None:
        figsize = (ncols * 3, nrows * 5)
        
    # wrapper for plt.subplots to avoid having to set graphical pars for stamp plots every time
    fig, axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize, sharex = True, sharey = True, subplot_kw = { "projection" : rg.crs["uk_map"]}, dpi= 100, facecolor='w', edgecolor='k', **kwargs)
    
    # for all suplots set same geographical extent, hide axes
    for axz in fig.axes:
        axz.set_extent((-2e5, 7e5, -1e5, 12.2e5), crs = rg.crs["uk_map"])
        axz.set_axis_off()
        #axz.coastlines()
    
    return fig, axs


def qmap(da, ax = None, colorbar = True, coastlines = True, cmap = None, vmin = None, vmax = None, label = "", title = "", title_kw = None, contour = False, contour_interval = 0.5, contour_offset = False, cities = None, extend = "neither", **kwargs):
    
    # Wrapper to quickly add a pcolormesh to existing axes
        
    if vmin is None: vmin = vrange(da)["vmin"]
    if vmax is None: vmax = vrange(da)["vmax"]
    
    if cmap is None:
        if (np.sign(vmin) != 0) and (np.sign(vmax) != 0) and (np.sign(vmin) != np.sign(vmax)):
            cmap = "RdBu_r"
        else:
            cmap = "viridis"
    
    if ax is None:
        fig, ax = ukmap_subplots()
    
    if contour:
        if contour_offset:
            if np.sign(vmin) == -1 and np.sign(vmax) == 1:
                halfrange = np.arange(contour_interval / 2, np.ceil(vmax) + contour_interval / 2, contour_interval)
                contour_levels = np.unique(np.concatenate([-halfrange[::-1], halfrange]))
            else:
                contour_levels = np.arange(np.floor(vmin) - (contour_interval/2), np.ceil(vmax) + (contour_interval/2), contour_interval)
        else:
            if np.sign(vmin) == -1 and np.sign(vmax) == 1:
                halfrange = np.arange(0, np.ceil(vmax) + contour_interval, contour_interval)
                contour_levels = np.unique(np.concatenate([-halfrange[::-1], halfrange]))
            else:
                contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + contour_interval, contour_interval)
        cbar = ax.contourf(da.projection_x_coordinate, da.projection_y_coordinate, da, contour_levels, vmin = vmin, vmax = vmax, cmap = cmap, **kwargs)
    else:
        cbar = ax.pcolormesh(rg.xv, rg.yv, da, vmin = vmin, vmax = vmax, cmap = cmap, **kwargs)
    ax.set_title(title, fontsize = "medium")
    
    if coastlines:
        ax.coastlines()
        
    if cities is not None:
        for city in cities:
            ax.plot(*rg.osgb_coords(city), marker = "x", ms = 4, color = "black")
    
    if colorbar:
        plt.colorbar(cbar, ax = ax, location = "bottom", pad = 0.01, label = label, extend = extend)
    
    return cbar


def add_city(cities, ax = None, marker = "x", ms = 4, color = "black", **kwargs):
    
    # eg: add_city("Bradford", ax = plt.gcf().axes[4])
    if type(cities) != list: cities = [cities]
    if ax is None: ax = plt.gca()
    for city in cities:
            ax.plot(*rg.osgb_coords(city), marker = marker, ms = ms, color = color, **kwargs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLOTS FOR SPECIFIC OUTPUTS







def biases(varnm, season, period = "19890101-20081231", dp = 1, rel_diff = False):
    
    # return biases
    obs = xr.open_mfdataset("/data/met/hadUK-grid/clim/"+varnm+"_hadukgrid_uk_12km_*_seasonal-climatology_"+period+".nc").sel(season = season)[varnm].load()
    
    # load all model runs, apply HadUK-grid land-sea mask
    rcms = xr.open_mfdataset("/data/met/ukcordex/*/*/*/clim/"+varnm+"_12km_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    gcms = xr.open_mfdataset("/data/met/cmip5/*/*/clim/"+varnm+"_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    ukcp = xr.open_mfdataset("/data/met/ukcp18/*/clim/"+varnm+"_rcp85_ukcp18_12km_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    
    gcms = gcms.assign_coords(run = [[gcm for gcm in list(gcm_markers.keys()) if n in gcm][0] for n in gcms.run.values])
    # split RCMs into evaluation & GCM-driven runs
    evals = rcms.sel(run = ["ECMWF" in rn for rn in rcms.run.values])
    rcms = rcms.sel(run = [not "ECMWF" in rn for rn in rcms.run.values])
    
    if rel_diff:
        biases = [((runs - obs) / obs * 100).mean(["projection_x_coordinate", "projection_y_coordinate"]).to_dataframe() for runs in [gcms, rcms, ukcp, evals]]
    else:
        biases = [(runs - obs).mean(["projection_x_coordinate", "projection_y_coordinate"]).to_dataframe() for runs in [gcms, rcms, ukcp, evals]]
    
    return biases


def rel_sds(varnm, season, period = "19890101-20081231", dp = 1):
    
    # return biases
    obs = xr.open_mfdataset("/data/met/hadUK-grid/clim/"+varnm+"_hadukgrid_uk_12km_*_seasonal-climatology_"+period+".nc").sel(season = season)[varnm].load()
    
    # load all model runs, apply HadUK-grid land-sea mask
    rcms = xr.open_mfdataset("/data/met/ukcordex/*/*/*/clim/"+varnm+"_12km_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    gcms = xr.open_mfdataset("/data/met/cmip5/*/*/clim/"+varnm+"_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    ukcp = xr.open_mfdataset("/data/met/ukcp18/*/clim/"+varnm+"_rcp85_ukcp18_12km_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    
    gcms = gcms.assign_coords(run = [[gcm for gcm in list(gcm_markers.keys()) if n in gcm][0] for n in gcms.run.values])
    evals = rcms.sel(run = ["ECMWF" in rn for rn in rcms.run.values])
    rcms = rcms.sel(run = [not "ECMWF" in rn for rn in rcms.run.values])
    
    sds = [(da.std(["projection_y_coordinate", "projection_x_coordinate"]) / obs.std(["projection_y_coordinate", "projection_x_coordinate"])).to_dataframe() for da in [gcms, rcms, ukcp, evals]]
    
    return sds


def split_biases(varnm, season, period = "19890101-20081231", rel_diff = False):
    
    gcm_bias, rcm_bias, ukcp_bias, eval_bias = biases(varnm = varnm, season = season, period = period, rel_diff = rel_diff)
    
    rcm_bias = pd.concat([rcm_bias, eval_bias])
    rcm_bias["gcm"] = rcm_bias.index.str.replace("p1_.+","p1", regex = True)
    rcm_bias["rcm"] = rcm_bias.index.str.replace(".+_","", regex = True)
    
    fig, axs = plt.subplots(1,4, figsize = (12,7), dpi = 100, sharey = True, gridspec_kw = {'width_ratios' : [0.25,1,1,0.25]})
    rcm_bias.boxplot(column = varnm, by = "rcm", grid = False, rot = 90, ax = axs[2], color = "black", showfliers = False)
    rcm_bias.boxplot(column = varnm, by = "gcm", grid = False, rot = 90, ax = axs[1], color = "black", showfliers = False)
    gcm_bias.boxplot(column = varnm, grid = False, ax = axs[0], color = "black", showfliers = False)
    ukcp_bias.boxplot(column = varnm, grid = False, ax = axs[3], color = "black", showfliers = False)

    mscatter([sorted(list(set(rcm_bias.rcm))).index(r) + 1 for r in rcm_bias.rcm], rcm_bias[varnm], ax = axs[2], m = [gcm_markers[g] for g in rcm_bias.gcm], c = [rcm_colours[r] for r in rcm_bias.rcm], zorder = 9, edgecolor = "black", s = 60)
    mscatter([sorted(list(set(rcm_bias.gcm))).index(g) + 1 for g in rcm_bias.gcm], rcm_bias[varnm], ax = axs[1], m = [gcm_markers[g] for g in rcm_bias.gcm], c = [rcm_colours[r] for r in rcm_bias.rcm], zorder = 9, edgecolor = "black", s = 60)
    mscatter(np.repeat(1,len(gcm_bias)), gcm_bias[varnm], ax = axs[0], m = [gcm_markers[g] for g in gcm_bias.index], c = "black", zorder = 9, edgecolor = "black", s = 60)
    mscatter(np.repeat(1,len(ukcp_bias)), ukcp_bias[varnm], ax = axs[3], m = "+", c = "black", zorder = 9, edgecolor = "black", s = 60)

    axs[2].set_title("UKCP bias")
    axs[2].set_title("RCM bias grouped by RCM")
    axs[1].set_title("RCM bias grouped by GCM")
    axs[0].set_title("GCM bias")
    
    for ax in axs:
        ax.axhline(0, linestyle = "--", color = "darkred")
        ax.set_xticklabels("")
        
    fig.suptitle("Bias in "+season + " "+varnm)
    gcm_legend(loc = 'upper left', bbox_to_anchor = (0.91, 0.93))
    rcm_legend(loc = 'lower left', bbox_to_anchor = (0.91, 0.15))
   
    

def stamp_plot(varnm, season, period = "19890101-20081231", ensemble = "EuroCORDEX", bias = True, cmap = None, save = False, title = False):
    
    # load the data
    data = load_eval(varnm, season, period)
    obs = list(data.values())[0]
    ens_list = list(data.values())[1:]
    ens_names = list(data.keys())[1:]
    
    if bias:
        ens_list = [ens - obs for ens in ens_list]
        b = "bias-"
        bt = "Bias in climatology"
        ct = "Bias in "+lc(var_cv[varnm]["plot_label"])
    else:
        ens_list = [ens.where(~np.isnan(obs)) for ens in ens_list]
        b = ""
        bt = "Climatology"
        ct = var_cv[varnm]["plot_label"]
    
    if not cmap: cmap = var_cv[varnm]["cmap"]
    
    if ensemble == "CMIP5":
        
        # Assuming no GCM data exists for EC-EARTH r3i1p1: will need to rejig if it becomes available
        fig, axs = ukmap_subplots(ncols = 9, nrows = 2, figsize = (16,6))
        fig.subplots_adjust(hspace = 0.25, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title: 
            fig.subplots_adjust(top = 0.85)
            fig.suptitle(bt+" of "+season+" "+varnm+" for CMIP5 GCMs", fontweight = "bold", y = 1)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("CMIP5-13")], ens_list[ens_names.index("CMIP5-EC")]], "new"))
        
        for gcm_nm in cmip5_ec:
            if gcm_nm in ens_list[ens_names.index("CMIP5-EC")].run.values:
                i = [nm for nm in cmip5_ec if not nm == "EC-EARTH_r3i1p1"].index(gcm_nm)
                gcm = ens_list[ens_names.index("CMIP5-EC")].sel(run = gcm_nm)
                qmap(gcm, ax = fig.axes[i], cmap = cmap, **vlims, colorbar = False, title = re.sub("_", "\n", gcm_nm))
        
        for gcm_nm in [nm for nm in cmip5_13 if not nm in cmip5_ec]:
            if gcm_nm in ens_list[ens_names.index("CMIP5-13")].run.values:
                i = [nm for nm in cmip5_13 if not nm in cmip5_ec].index(gcm_nm)
                gcm = ens_list[ens_names.index("CMIP5-13")].sel(run = gcm_nm)
                cbar = qmap(gcm, ax = fig.axes[i+9], cmap = cmap, **vlims, colorbar = False, title = re.sub("_", "\n", gcm_nm))
                
        axs[0,0].text(-0.07, 0.55, "EuroCORDEX GCMs", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        axs[1,0].text(-0.07, 0.55, "CMIP5-13 GCMs", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[1,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = ct)
    
    elif ensemble[:4] == "UKCP":
        
        fig, axs = ukmap_subplots(ncols = 12, nrows = 2, figsize = (20,6))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title: 
            fig.subplots_adjust(top = 0.93)
            fig.suptitle(bt+" of "+season+" "+varnm+" for UKCP18 global & regional runs", fontweight = "bold", y = 1)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("UKCP18 60km")], ens_list[ens_names.index("UKCP18 12km")]], "new"))
        
        for i in range(12):
            cbar = qmap(ens_list[ens_names.index("UKCP18 60km")].isel(run = i), ax = axs[0,i], cmap = cmap, **vlims, colorbar = False, title = ens_list[ens_names.index("UKCP18 12km")].isel(run = i).run.values.tolist())
            qmap(ens_list[ens_names.index("UKCP18 12km")].isel(run = i), ax = axs[1,i], cmap = cmap, **vlims, colorbar = False)
            
        axs[0,0].text(-0.07, 0.55, "60km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        axs[1,0].text(-0.07, 0.55, "12km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[1,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = ct)
    
    else:
        
        # Otherwise, plot all the EuroCORDEX runs, plus their driving GCMs
        fig, axs = ukmap_subplots(ncols = 11, nrows = 11, figsize = (26,24))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title:
            fig.suptitle(bt+" of "+season+" "+varnm+" for EuroCORDEX RCMs and CMIP5 driving RCMs", fontweight = "bold", y = 1)
            fig.subplots_adjust(top = 0.96)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("CMIP5-EC")], ens_list[ens_names.index("EuroCORDEX")], ens_list[ens_names.index("ERA-EuroCORDEX")]], "new"))
            
        # could filter out any unwanted GCMs here as well
        cmip_gcms = [x for x in list(gcm_full_names.keys())]
        ec_gcms = [x for x in list(gcm_full_names.values())]
            
        for gcm_nm in cmip5_ec:
            if gcm_nm in ens_list[ens_names.index("CMIP5-EC")].run.values:
                cbar = qmap(ens_list[ens_names.index("CMIP5-EC")].sel(run = gcm_nm), ax = axs[0,cmip_gcms.index(gcm_nm)+1], cmap = cmap, **vlims, colorbar = False)
        
        for rcm_nm in ens_list[ens_names.index("EuroCORDEX")].run.values:
            rcm_i = list(rcm_colours.keys()).index(re.sub(".+_", "", rcm_nm))
            gcm_i = ec_gcms.index(re.sub("p1_.+","p1",rcm_nm))
            qmap(ens_list[ens_names.index("EuroCORDEX")].sel(run = rcm_nm), ax = axs[rcm_i+1,gcm_i+1], cmap = cmap, **vlims, colorbar = False)
    
        for rcm_nm in ens_list[ens_names.index("ERA-EuroCORDEX")].run.values:
            rcm_i = list(rcm_colours.keys()).index(re.sub(".+_", "", rcm_nm))
            qmap(ens_list[ens_names.index("ERA-EuroCORDEX")].sel(run = rcm_nm), ax = axs[rcm_i+1,0], cmap = cmap, **vlims, colorbar = False)
        
        # add GCM & RCM names to first row & column. Currently impossible to add a ylabel to cartopy axes.
        for i in range(10): axs[0,i+1].set_title(re.sub("_", "\n",cmip_gcms[i]))
        for j in range(10): axs[j+1,0].text(-0.07, 0.55, list(rcm_colours.keys())[j], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[j+1,0].transAxes, fontsize = "large")
        axs[0,0].text(-0.07, 0.55, "GCM", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        axs[0,0].set_title("ECMWF-ERAINT")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.01, pad = 0.01, label = ct)
    
    if save: plt.savefig("/data/met/reports/evaluation/plots/"+varnm+"_"+season+"_"+period+"_"+ensemble.lower()+"_"+b+"stampplot.png")
    


def orog_correlation(varnm, season, period = "19890101-20081231"):
    
    obs = xr.open_mfdataset("/data/met/hadUK-grid/clim/"+ varnm +"_*"+period+"*")[varnm].sel(season = season).load()
    orog = xr.open_dataset("/data/met/processing/01_maps/orog_land-rcm_uk_12km_osgb.nc").surface_altitude.where(~np.isnan(obs))
    
    rcms = xr.open_mfdataset("/data/met/ukcordex/*/*/*/clim/"+varnm+"_12km_*_seasonal-climatology_"+period+".nc").sel(season = season).where(~np.isnan(obs))[varnm].load()
    
    sds = rcms.std(["projection_x_coordinate", "projection_y_coordinate"]).to_dataframe()    
    corrs = xr.corr(rcms, orog, ["projection_x_coordinate", "projection_y_coordinate"]).rename(varnm).to_dataframe()
    
    corrs["rcm"] = [re.sub(".+_","", rn) for rn in corrs.index.values]
    corrs["gcm"] = [re.sub("p1_.+","p1", rn) for rn in corrs.index.values]
    
    fig, axs = plt.subplots(figsize = (7,5), dpi= 100, facecolor='w', edgecolor='k')
    
    mscatter(sds[varnm], corrs[varnm], ax = axs, m = [gcm_markers[rn] for rn in corrs.gcm], c = [rcm_colours[rn] for rn in corrs.rcm], edgecolor = "black", s = 60)
    axs.set_xlabel("SD of climatological pattern")
    axs.set_ylabel("Correlation of SD with orography")

    

    
def res_boxplot(varnm, season, pca = True, period = "19890101-20081231", design = "allruns", hlines = None, by = "gcm", ylim = None, xlabels = True):
    
    # load the runs
    obs = xr.open_mfdataset("/data/met/hadUK-grid/clim/"+varnm+"*_seasonal-climatology_"+period+".nc")[varnm].sel(season = season).load()
    err = xr.open_mfdataset("/data/met/ukcordex/*/*/*/clim/"+varnm+"_*_seasonal-climatology_"+period+".nc")[varnm].sel(season = season).load() - obs
    ukcp = xr.open_mfdataset("/data/met/ukcp18/*/clim/"+varnm+"_*"+period+".nc")[varnm].sel(season = season).load() - obs
    
    # load PCA, compute field explained by mean + first two components of GCM & RCM
    svd = xr.open_dataset("/data/met/processing/80_results/evaluation-pca/pca_"+varnm+"_"+period+"_"+design+".nc").sel(season = season).isel(pc = slice(0,5))
        
    if pca:
        expl = (svd.svecs * svd.svals * svd.scores).sum("pc", skipna = False).rename("explained")
        res = (err - expl).rename("residual")
        ukcp_res = (ukcp - expl).rename("residual")
        st = " after first two GCM & RCM components accounted for"
    else:
        res = err.rename("residual").sel(run = np.isin(err.run, svd.run))
        ukcp_res = ukcp.rename("residual")
        st = ""
    
    if by == "gcm":
        
        # tidy up the GCM list for plotting
        gcms = sorted(list(set([re.sub("p1_.+", "p1", rn) for rn in res.run.values])))
        if "ECMWF-ERAINT_r1i1p1" in gcms:
            gcms = ["ECMWF-ERAINT_r1i1p1"] + [g for g in gcms if not "ECMWF" in g]
            
        fig, axs = plt.subplots(ncols = (len(gcms)+1), figsize = (len(gcms)*2,4), dpi = 100, sharey = True)
        plt.subplots_adjust(top = 0.85)
        
        for i in range(len(gcms)):
            axz = axs[i]
            rs = rg.map2vec(res.sel(run = [re.sub("p1_.+", "p1", rn) == gcms[i] for rn in res.run.values]))
            bplot = axz.boxplot(rs.values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
            axz.set_title(re.sub("_","\n",gcms[i]), fontsize = "small")
            axz.set_xticklabels(rotation = 90, labels = [re.sub(".+_","",rn) for rn in rs.run.values])
            for patch, color in zip(bplot['boxes'], [rcm_colours[re.sub(".+_", "", rn)] for rn in rs.run.values]):
                patch.set_facecolor(color)
        
        axz = axs[len(gcms)]
        axz.set_title("UKCP18\n", fontsize = "small")
        bplot = axz.boxplot(rg.map2vec(ukcp_res).values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
        for patch, color in zip(bplot['boxes'], [rcm_colours["HadREM3-GA7-05"]] + list(np.repeat("White", 11))):
            patch.set_facecolor(color)
    
    else:
        rcms = sorted(list(set([re.sub(".+_", "", rn) for rn in res.run.values])))
        
        fig, axs = plt.subplots(ncols = (len(rcms)+1), figsize = (len(rcms)*2,4), dpi = 100, sharey = True)
        plt.subplots_adjust(top = 0.85)
        
        for i in range(len(rcms)):
            axz = axs[i]
            rs = rg.map2vec(res.sel(run = [re.sub(".+_", "", rn) == rcms[i] for rn in res.run.values]))
            bplot = axz.boxplot(rs.values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
            axz.set_xticklabels(rotation = 90, labels = [re.sub("p1_.+","p1",rn) for rn in rs.run.values])
            axz.set_title(rcms[i], fontsize = "small")
            for patch, color in zip(bplot['boxes'], [rcm_colours[re.sub(".+_", "", rn)] for rn in rs.run.values]):
                patch.set_facecolor(color)
        
        axz = axs[len(rcms)]
        axz.set_title("UKCP18\n", fontsize = "small")
        bplot = axz.boxplot(rg.map2vec(ukcp_res).values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
        axz.set_xticklabels("")
        for patch, color in zip(bplot['boxes'], [rcm_colours["HadREM3-GA7-05"]] + list(np.repeat("White", 11))):
            patch.set_facecolor(color)
            
    for axz in axs:
        axz.axhline(0, linestyle = "--", color = "darkred")
        if hlines:
            for hl in hlines: axz.axhline(hl, linestyle = "--", color = "grey", alpha = 0.5, linewidth = 1)
        if ylim: axz.set_ylim(*ylim)
        if not xlabels: axz.set_xticklabels("")
        
    fig.suptitle("Residuals of "+ lc(var_cv[varnm]["plot_label"]) + st, fontweight = "bold")
    


def design_effect(varnm, season, pca = True, period = "19890101-20081231", hlines = None, by = "gcm", ylim = None, xlabels = True, showall = False):
    
    # boxplot of change when using balanced vs unbalanced runs

    # load PCAs, compute explained fields
    u_pca = xr.open_dataset("/data/met/processing/80_results/evaluation-pca/pca_"+varnm+"_"+period+"_allruns.nc").sel(season = season).isel(pc = slice(0,5))
    b_pca = xr.open_dataset("/data/met/processing/80_results/evaluation-pca/pca_"+varnm+"_"+period+"_balanced.nc").sel(season = season).isel(pc = slice(0,5))
    
    u_expl = (u_pca.svecs * u_pca.svals * u_pca.scores).sum("pc", skipna = False).rename("exp")
    b_expl = (b_pca.svecs * b_pca.svals * b_pca.scores).sum("pc", skipna = False).rename("exp")
    
    e_diff = b_expl - u_expl
    ukcp_diff = e_diff.sel(run = np.isin(e_diff.run, [rn for rn in e_diff.run.values if not "_" in rn]))
    ec_diff = e_diff.sel(run = np.isin(e_diff.run, [rn for rn in e_diff.run.values if "_" in rn]))
    
    if by == "gcm":
        # tidy up the GCM list for plotting
        if showall:
            gcms = sorted(list(set([re.sub("p1_.+", "p1", rn) for rn in u_expl.run.values if "p1_" in rn])))
        else:
            gcms = sorted(list(set([re.sub("p1_.+", "p1", rn) for rn in ec_diff.run.values])))
        if "ECMWF-ERAINT_r1i1p1" in gcms:
            gcms = ["ECMWF-ERAINT_r1i1p1"] + [g for g in gcms if not "ECMWF" in g]
            
        fig, axs = plt.subplots(ncols = (len(gcms)+1), figsize = (len(gcms)*3,4), dpi = 100, sharey = True)
        plt.subplots_adjust(top = 0.85)
        
        for i in range(len(gcms)):
            axz = axs[i]
            axz.set_title(re.sub("_","\n",gcms[i]), fontsize = "small")
            if gcms[i] in [re.sub("p1_.+", "p1", rn) for rn in ec_diff.run.values]:
                rs = rg.map2vec(ec_diff.sel(run = [re.sub("p1_.+", "p1", rn) == gcms[i] for rn in ec_diff.run.values]))
                bplot = axz.boxplot(rs.values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
                axz.set_xticklabels(rotation = 90, labels = [re.sub(".+_","",rn) for rn in rs.run.values])
                for patch, color in zip(bplot['boxes'], [rcm_colours[re.sub(".+_", "", rn)] for rn in rs.run.values]):
                    patch.set_facecolor(color)
            else:
                axz.set_xticklabels("")
        
        axz = axs[len(gcms)]
        axz.set_title("UKCP18\n", fontsize = "small")
        bplot = axz.boxplot(rg.map2vec(ukcp_diff).values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
        for patch, color in zip(bplot['boxes'], [rcm_colours["HadREM3-GA7-05"]] + list(np.repeat("White", 11))):
            patch.set_facecolor(color)
            
    else: 
        if showall:
            rcms = sorted(list(set([re.sub(".+_", "", rn) for rn in u_expl.run.values if "p1_" in rn])))
        else:
            rcms = sorted(list(set([re.sub(".+_", "", rn) for rn in ec_diff.run.values])))
        
        fig, axs = plt.subplots(ncols = (len(rcms)+1), figsize = (len(rcms)*2,4), dpi = 100, sharey = True)
        plt.subplots_adjust(top = 0.85)
        
        for i in range(len(rcms)):
            axz = axs[i]
            axz.set_title(rcms[i], fontsize = "small")
            if rcms[i] in [re.sub(".+_", "", rn) for rn in ec_diff.run.values]:
                rs = rg.map2vec(ec_diff.sel(run = [re.sub(".+_", "", rn) == rcms[i] for rn in ec_diff.run.values]))
                bplot = axz.boxplot(rs.values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
                axz.set_xticklabels(rotation = 90, labels = [re.sub("p1_.+","p1",rn) for rn in rs.run.values])
                for patch, color in zip(bplot['boxes'], [rcm_colours[re.sub(".+_", "", rn)] for rn in rs.run.values]):
                    patch.set_facecolor(color)
            else:
                axz.set_xticklabels("")
            
        axz = axs[len(rcms)]
        axz.set_title("UKCP18\n", fontsize = "small")
        bplot = axz.boxplot(rg.map2vec(ukcp_diff).values.transpose(), patch_artist = True, flierprops = {"marker" : "o", "markersize" : 1}, medianprops = {"color" : "black"})
        axz.set_xticklabels("")
        for patch, color in zip(bplot['boxes'], [rcm_colours["HadREM3-GA7-05"]] + list(np.repeat("White", 11))):
            patch.set_facecolor(color)
    
    for axz in axs:
        axz.axhline(0, linestyle = "--", color = "darkred")
        if hlines:
            for hl in hlines: axz.axhline(hl, linestyle = "--", color = "grey", alpha = 0.5, linewidth = 1)
        if ylim: axz.set_ylim(*ylim)
        if not xlabels: axz.set_xticklabels("")
        
    fig.suptitle("Change in explained "+ lc(var_cv[varnm]["plot_label"]) + " when using balanced rather than unbalanced design", fontweight = "bold")
    
    
    
    

        
        
    
def bias_sdmaps(varnm, season, cmap = None, vmin = 0, vmax = None, contour = False, contour_interval = 1, save = False, title = False, extend = "neither"):
    
    sds = xr.open_dataset("/data/met/processing/80_pca-anova/bias-pca/allvars_bias-sd.nc").sel(season = season, varnm = varnm).sd
    
    if cmap is None:
        if var_cv[varnm]["cmap"] == "RdBu":
            cmap = "Blues"
        elif var_cv[varnm]["cmap"] == "RdBu_r":
            cmap = "Reds"
        else:
            cmap = "plasma"
    
    if vmax is None:
        vmax = sds.max()
    
    fig, axs = ukmap_subplots(ncols = 6, figsize = (6*3 - 2,5))
    
    for i in range(len(sds.ens_name)):
        cbar = qmap(sds.isel(ens_name = i), ax = axs[i], colorbar = False, title = "("+string.ascii_lowercase[i]+") "+sds.ens_name.values[i], 
                    vmin = vmin, vmax = vmax, cmap = cmap, contour = contour, contour_interval = contour_interval) 
    
    plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.035, pad = 0.01, label = "Standard deviation of "+lc(var_cv[varnm]["plot_label"]), extend = extend)
    
    if title:
        fig.suptitle("Standard deviation of biases in "+season+" "+varnm+" in each ensemble", fontweight = "bold")
    
    if save: plt.savefig("/data/met/reports/evaluation/plots/"+varnm+"_"+season+"_19890101-20081231_sd-maps.png", bbox_inches='tight')
    

############################################################################################################################################################################
############################################################################################################################################################################

            
            
def change_stampplots(varnm, season, period = "20491201-20791130", ensemble = "EuroCORDEX", rel_diff = False, recover_clim = False, cmap = None, save = False, title = False):
    
    # load the data
    data = load_diff(varnm = varnm, season = season, period = period, rel_diff = rel_diff, recover_clim = recover_clim)
    ens_list = list(data.values())
    ens_names = list(data.keys())
    
    if rel_diff:
        main = "Relative projected change in "+season+" " + lc(var_cv[varnm]["plot_label"])+" over the UK between 1980-2010 and "+period[:4]+"-"+period[9:13]
        cbar_label = "% change in " + lc(var_cv[varnm]["plot_label"])
        d = ""
    elif recover_clim:
        main = "Average "+season+" "+varnm+" over the UK from "+period[:4]+"-"+period[9:13]
        cbar_label = var_cv[varnm]["plot_label"]
        d = ""
    else:
        main = "Projected change in "+season+" " + lc(var_cv[varnm]["plot_label"])+" over the UK between 1980-2010 and "+period[:4]+"-"+period[9:13]
        cbar_label = "Change in " + lc(var_cv[varnm]["plot_label"])
        d = ""
    
    if not cmap: cmap = var_cv[varnm]["cmap"]
    
    if ensemble == "CMIP5":
        
        # Assuming no GCM data exists for EC-EARTH r3i1p1: will need to rejig if it becomes available
        fig, axs = ukmap_subplots(ncols = 9, nrows = 2, figsize = (16,6))
        fig.subplots_adjust(hspace = 0.25, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title: 
            fig.subplots_adjust(top = 0.85)
            fig.suptitle(main+" for CMIP5 GCMs", fontweight = "bold", y = 1)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("CMIP5-13")], ens_list[ens_names.index("CMIP5-EC")]], "new"))
        
        for gcm_nm in cmip5_ec:
            if gcm_nm in ens_list[ens_names.index("CMIP5-EC")].run.values:
                i = [nm for nm in cmip5_ec if not nm == "EC-EARTH_r3i1p1"].index(gcm_nm)
                gcm = ens_list[ens_names.index("CMIP5-EC")].sel(run = gcm_nm)
                qmap(gcm, ax = fig.axes[i], cmap = cmap, **vlims, colorbar = False, title = re.sub("_", "\n", gcm_nm))
        
        for gcm_nm in [nm for nm in cmip5_13 if not nm in cmip5_ec]:
            if gcm_nm in ens_list[ens_names.index("CMIP5-13")].run.values:
                i = [nm for nm in cmip5_13 if not nm in cmip5_ec].index(gcm_nm)
                gcm = ens_list[ens_names.index("CMIP5-13")].sel(run = gcm_nm)
                cbar = qmap(gcm, ax = fig.axes[i+9], cmap = cmap, **vlims, colorbar = False, title = re.sub("_", "\n", gcm_nm))
                
        axs[0,0].text(-0.07, 0.55, "EuroCORDEX GCMs", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        axs[1,0].text(-0.07, 0.55, "CMIP5-13 GCMs", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[1,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = cbar_label)
    
    elif ensemble[:4] == "UKCP":
        
        fig, axs = ukmap_subplots(ncols = 12, nrows = 2, figsize = (20,6))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title: 
            fig.subplots_adjust(top = 0.93)
            fig.suptitle(main+" for UKCP18 global & regional runs", fontweight = "bold", y = 1)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("UKCP18 60km")], ens_list[ens_names.index("UKCP18 12km")]], "new"))
        
        for i in range(12):
            cbar = qmap(ens_list[ens_names.index("UKCP18 60km")].isel(run = i), ax = axs[0,i], cmap = cmap, **vlims, colorbar = False, title = ens_list[ens_names.index("UKCP18 12km")].isel(run = i).run.values.tolist())
            qmap(ens_list[ens_names.index("UKCP18 12km")].isel(run = i), ax = axs[1,i], cmap = cmap, **vlims, colorbar = False)
            
        axs[0,0].text(-0.07, 0.55, "60km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        axs[1,0].text(-0.07, 0.55, "12km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[1,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = cbar_label)
    
    else:
        
        # Otherwise, plot all the EuroCORDEX runs, plus their driving GCMs
        fig, axs = ukmap_subplots(ncols = 10, nrows = 11, figsize = (20,24))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title:
            fig.suptitle(main+" for EuroCORDEX RCMs and CMIP5 driving RCMs", fontweight = "bold", y = 1)
            fig.subplots_adjust(top = 0.96)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("CMIP5-EC")], ens_list[ens_names.index("EuroCORDEX")]], "new"))
        
        for gcm_nm in cmip5_ec:
            if gcm_nm in ens_list[ens_names.index("CMIP5-EC")].run.values:
                cbar = qmap(ens_list[ens_names.index("CMIP5-EC")].sel(run = gcm_nm), ax = axs[0,cmip5_ec.index(gcm_nm)], cmap = cmap, **vlims, colorbar = False)
        
        for rcm_nm in ens_list[ens_names.index("EuroCORDEX")].run.values:
            rcm_i = list(rcm_colours.keys()).index(re.sub(".+_", "", rcm_nm))
            gcm_i = list(gcm_markers.keys()).index(re.sub("p1_.+","p1",rcm_nm))
            qmap(ens_list[ens_names.index("EuroCORDEX")].sel(run = rcm_nm), ax = axs[rcm_i+1,gcm_i-1], cmap = cmap, **vlims, colorbar = False)
            
        # add GCM & RCM names to first row & column. Currently impossible to add a ylabel to cartopy axes.
        for i in range(10): axs[0,i].set_title(re.sub("_", "\n",cmip5_ec[i]))
        for j in range(10): axs[j+1,0].text(-0.07, 0.55, list(rcm_colours.keys())[j], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[j+1,0].transAxes, fontsize = "large")
        axs[0,0].text(-0.07, 0.55, "GCM", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.01, pad = 0.01, label = cbar_label)
    
    if save: plt.savefig("/data/met/reports/evaluation/plots/"+varnm+"_"+season+"_"+period+"_"+ensemble.lower()+"_"+b+"stampplot.png")
        
        
        

def xyline(x, y, ax = None, **kwargs):
    
    # method to quickly add line of best fit through points
    theta = np.polyfit(x, y, 1)
    y_line = theta[1] + theta[0] * np.sort(x)
    
    if ax is None:
        plt.plot(np.sort(x), y_line, **kwargs)
    else:
        ax.plot(np.sort(x), y_line, **kwargs)
    
    
    
def qboxplot(da_dict, model_legend = False, j = 0.025, ax = None, biaslines = "auto", blwidth = 1, hline = 0, legend_pos = "best"):
    
    # eg qboxplot(load_diff("pr", "DJF", "20191201-20491130"))
    
    ens_list = list(da_dict.values())
    ens_names = list(da_dict.keys())  
    means = [ens.mean(["projection_x_coordinate", "projection_y_coordinate"]) for ens in ens_list]
          
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (12,6), dpi= 100, facecolor='w', edgecolor='k')
    
    ax.violinplot(means, showextrema = False)
    ax.boxplot(means, labels = [re.sub("ERA-EuroCORDEX","ERA-RCM",n) for n in ens_names], meanline = True, showmeans = True, zorder = 3, 
               meanprops = {"color" : "black"}, medianprops = {"color" : "black"}, flierprops = {"marker" : ""})
    
    offsets = dict(zip(list(gcm_markers.keys())[1:11], list(np.arange(j*-4.5, j*5.5, j).round(3))))
    
    if "CMIP5-13" in ens_names:
        i = ens_names.index("CMIP5-13")
        mscatter([offsets[gcm_full_names[n]]+i+1 if n in list(gcm_full_names.keys()) else 0+i+1 for n in means[i].run.values], means[i],
                 m = [gcm_markers[mdl] for mdl in means[i].run.values], color = gcm_colours(means[i].run.values), edgecolor = "k", s = 50, zorder = 5, ax = ax)
    
    if "CMIP5-EC" in ens_names:
        i = ens_names.index("CMIP5-EC")
        mscatter([offsets[gcm_full_names[n]]+i+1 for n in means[i].run.values], means[i],
                 m = [gcm_markers[mdl] for mdl in means[i].run.values], color = gcm_colours(means[i].run.values), edgecolor = "k", s = 50, zorder = 5, ax = ax)
    
    if "EuroCORDEX" in ens_names:
        i = ens_names.index("EuroCORDEX")
        rcm_gcms = [re.sub("p1_.+", "p1", rn) for rn in means[i].run.values]
        rcm_rcms = [re.sub(".+_", "", rn) for rn in means[i].run.values]
        mscatter([offsets[n]+i+1 for n in rcm_gcms], means[i],
                 color = [rcm_colours[n] for n in rcm_rcms], m = [gcm_markers[n] for n in rcm_gcms], edgecolor = "k", zorder = 5, s = 50, ax = ax)
    
    if "ERA-EuroCORDEX" in ens_names:
        i = ens_names.index("ERA-EuroCORDEX")
        eval_rcms = [re.sub(".+_", "", rn) for rn in means[i].run.values]
        ax.scatter(np.repeat(i+1, len(means[i])), means[i], 
                   color = [rcm_colours[n] for n in eval_rcms], marker = gcm_markers["ECMWF-ERAINT_r1i1p1"], edgecolor = "k", zorder = 5, s = 50)
    
    if "UKCP18 60km" in ens_names:
        i = ens_names.index("UKCP18 60km")
        ax.scatter(np.repeat(i+1, len(means[i])), means[i], color = "black", marker = ".", edgecolor = "k", zorder = 5, s = 60)
        ax.scatter(i+1, means[i][0],
                   color = rcm_colours["HadREM3-GA7-05"], marker = ".", edgecolor = "k", zorder = 5, s = 60)
    
    if "UKCP18 12km" in ens_names:
        i = ens_names.index("UKCP18 12km")
        ax.scatter(np.repeat(i+1, len(means[i])), means[i], color = "black", marker = ".", edgecolor = "k", zorder = 5, s = 60)
        ax.scatter(i+1, means[i][0],
                   color = rcm_colours["HadREM3-GA7-05"], marker = ".", edgecolor = "k", zorder = 5, s = 60)
    
    if biaslines == "auto":
        bl_vals = np.array([v for v in np.concatenate([m.values for m in means]) if not np.isnan(v)])
        biaslines = [int(np.floor(bl_vals.min())), int(np.ceil(bl_vals.max()))]
    
    if biaslines is not None:
        for i in np.arange(biaslines[0], biaslines[1] + blwidth, blwidth):
            ax.axhline(i, zorder = -99, color = "grey", alpha = 0.5, linewidth = 0.5)
    
    if hline is not None:
        ax.axhline(hline, color = "darkred", ls = "--")
    
    # legend for boxplots
    bplot_handles = [matplotlib.lines.Line2D([], [], color = "black", linestyle = "-"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "--")]
    bplot_labels = ["Group median", "Group mean"]
    ax.legend(handles = bplot_handles, labels = bplot_labels, edgecolor = "white", loc = legend_pos)
    
    

############################################################################################################################################################################
############################################################################################################################################################################


def pca_plot(pca, cmap = None, contour = False, contour_interval = 1, markers = None, colours = None, fs = 12, cbar_label = "", vmin = None, vmax = None, xylims = None):
    
    # generic PCA plotting function to be used for biases & changes
    modes = pca.svecs * pca.svals
    
    # open fig first in order to add legend
    fig, axs = ukmap_subplots(ncols = 4, figsize = (13,4), gridspec_kw = {'width_ratios' : [1,1,1,2]})
    fig.subplots_adjust(top = 0.85, wspace = 0)
    fig.tight_layout()
    
    ec = "black"
    ms = 60
    # set plotting marker & colour
    if "i1p1" in pca.run.values[0]:
        # GCM effects
        markers = [gcm_markers[g] for g in [re.sub("p1_.+","p1",rn) for rn in pca.run.values]]
        colours = "white"
        gcm_legend(gcm_names = pca.run.values, short_names = True, ax = fig, loc = 'center left', bbox_to_anchor = (0.95, 0.5))
    elif "HadREM3-GA7-05" in pca.run.values:
        # RCM effects
        colours = [rcm_colours[r] for r in [re.sub(".+_","",rn) for rn in pca.run.values]]
        rcm_legend(rcm_names = pca.run.values, ax = fig, loc = 'center left', bbox_to_anchor = (0.95, 0.5))
        markers = "o"
    elif len(pca.run.values[0]) == 2:
        # UKCP ensemble
        if markers is None:
            markers = ["$" + r + "$" for r in pca.run.values]
            ec = [rcm_colours["HadREM3-GA7-05"] if r == "01" else "black" for r in pca.run.values]
            ms = 100
        colours = [rcm_colours["HadREM3-GA7-05"] if r == "01" else "black" for r in pca.run.values]
        ukcp_labels = ["Unperturbed", "Perturbed"]
        ukcp_handles = [matplotlib.lines.Line2D([], [], color = colours[i], marker = "o", markersize = 6, markeredgecolor = "black", linestyle = "None") for i in range(2)]
        fig.legend(handles = ukcp_handles, labels = ukcp_labels, edgecolor = "white", title = "Run type", loc = 'center left', bbox_to_anchor = (0.95, 0.5))
    else:
        # otherwise, use generic characters
        if markers is None: markers = "o"
        if colours is None: colours = "black"
    
    if not cmap: cmap = "RdBu_r"
        
    vlims = vrange(modes)
    if vmin is not None: vlims["vmin"] = vmin
    if vmax is not None: vlims["vmax"] = vmax

    map_args = { "colorbar" : False, "cmap" : fix_cmap(cmap, **vlims), "contour" : contour, "contour_interval" : contour_interval}
        
    ve = [" ("+str(int(x.round()))+"%)" for x in pca.var_expl]
    
    cbar = qmap(modes.sel(pc = "mean"), ax = fig.axes[0], **vlims, **map_args)
    fig.axes[0].set_title("Ensemble mean", size = fs)
    qmap(modes.sel(pc = "1"), ax = fig.axes[1], **vlims, **map_args)
    fig.axes[1].set_title("EPP1"+ve[1], size = fs)
    qmap(modes.sel(pc = "2"), ax = fig.axes[2], **vlims, **map_args)
    fig.axes[2].set_title("EPP2"+ve[2], size = fs)

    plt.colorbar(cbar, ax = fig.axes[0:3], location = "bottom", pad = 0.04, fraction = 0.05).set_label(label = cbar_label, size = fs)
    
    ax3 = fig.add_subplot(144)        
    mscatter(pca.scores.sel(pc = "1"), pca.scores.sel(pc = "2"), ax = ax3, m = markers, c = colours, s = ms, edgecolor = ec, zorder = 9)
        
    # axes & labels
    if xylims is None:
        score_range = vrange(pca.scores.sel(pc = ["1", "2"]).values * 1.2).values()
    else:
        score_range = xylims
        
    ax3.set_xlim(*score_range)
    ax3.set_ylim(*score_range)
    ax3.set_xlabel("First EPP score")
    ax3.set_ylabel("Second EPP score")
    ax3.set_title("Contribution from each pattern", fontsize = fs)
    ax3.axvline(0, linestyle = "--", color = "grey")
    ax3.axhline(0, linestyle = "--", color = "grey")
    ax3.set_aspect("equal", adjustable = "box")