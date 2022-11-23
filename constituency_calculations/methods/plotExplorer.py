import xarray as xr
import pandas as pd
import glob
import re

import matplotlib
import matplotlib.pyplot as plt

from taylorDiagram import TaylorDiagram
from misc import load_eval, load_diff, xy
from dicts import *
from plotting import *
from pc_decomposition import qpca, qpca_plot
from unbalanced_decomposition import *

# default to tight bounding box
matplotlib.rcParams['savefig.bbox'] = "tight"

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore", message = "FixedFormatter should only be used together with FixedLocator")  # something about adding vertical gridlines to variance explained


####################################################################################################################

all_slices = ["19890101-20081231"] + [str(ys)+"1201-"+str(ys+30)+"1130" for ys in [1980] + list(range(1989, 2059, 10))] + [str(x) for x in [1.5,2,3,4]]

####################################################################################################################

def tabulate_plots(varnm = None, res = None, period = None, season = None, details = False, by_period = False):
    
    df = pd.DataFrame([re.sub("\.png", "", re.sub(".+/", "", x)).split("_") for x in glob.glob("/data/met/plot-explorer/images/*/*.png")], columns = ["varnm", "res", "plottype", "period", "season"])
        
    if varnm: df = df[df.varnm == varnm]
    if res: df = df[df.res == res]
    if period: df = df[df.period == period]
    if season: df = df[df.season == season]
    
    if by_period:
        return pd.crosstab(index=df.varnm, columns=df.period)
    elif details:
        return df
    else:
        return pd.crosstab(index=df.varnm, columns=df.plottype)
    

####################################################################################################################

def load_data(varnm, period, season, verbose = True, trim = False):
        
    # for evaluation period, check that obs exist
    if period == "19890101-20081231" and len(glob.glob("/data/met/hadUK-grid/slices/"+varnm+"_*.nc")) == 0:
        if verbose: print("0", end = "")
        return
    
    # load all the slices
    runs = {"EuroCORDEX" : [xr.open_dataset(fnm).sel(season = season)[varnm] for fnm in glob.glob("/data/met/ukcordex/*/*/*/slices/"+varnm+"_12km_*_ts30.nc")],
            "gcms" : [xr.open_dataset(fnm).sel(season = season)[varnm] for fnm in glob.glob("/data/met/cmip5/*/*/slices/"+varnm+"_12km_*_ts30.nc")],
            "UKCP18 12km" : [xr.open_dataset(fnm).sel(season = season)[varnm] for fnm in sorted(glob.glob("/data/met/ukcp18/*/slices/"+varnm+"_12km_*_ts30.nc"))],
            "UKCP18 60km" : [xr.open_dataset(fnm).sel(season = season)[varnm] for fnm in sorted(glob.glob("/data/met/ukcp18/60km/*/slices/"+varnm+"_12km_*_ts30.nc"))]}
        
    # split EuroCORDEX runs into evaluation & GCM-driven
    runs["ERA-EuroCORDEX"] = [run for run in runs["EuroCORDEX"] if "ECMWF" in run.run.values[0]]
    runs["EuroCORDEX"] = [run for run in runs["EuroCORDEX"] if not "ECMWF" in run.run.values[0]]
    
    # split GCMs into two ensembles
    runs["CMIP5-13"] = [run for run in runs["gcms"] if run.run.values in cmip5_13]
    runs["CMIP5-EC"] = [run for run in runs["gcms"] if run.run.values in cmip5_ec]
    
    # concatenate ensembles & reorder
    runs = {k : xr.concat(v, "run") for k, v in runs.items() if len(v) > 0}
    runs = {k : runs[k] for k in ["CMIP5-13", "CMIP5-EC", "EuroCORDEX", "ERA-EuroCORDEX", "UKCP18 60km", "UKCP18 12km"] if k in list(runs.keys())}
    
    # set any infinite values to NA
    runs = {k : v.where(np.isfinite(v)) for k,v in runs.items()}
    
    # select the period of interest
    if period in ["19890101-20081231"]:
        
        # evaluation period: return bias
        obs = xr.open_mfdataset("/data/met/hadUK-grid/slices/"+varnm+"_*.nc")[varnm].sel(season = season).load().squeeze(drop = True)
        if "period" in obs.coords:
            obs = obs.sel(period = period)
        runs = {k : v.sel(period = period) - obs for k, v in runs.items()}
        runs["obs"] = obs
        if obs.mean() == 0: 
            if verbose: print("0", end = "")
            return
            
    elif period in ["1", "19801201-20101130"]:
        
        # reference period: return absolute value (excluding evaluation runs)
        runs = {k : v.sel(period = period) for k, v in runs.items() if not "ERA" in k}
        
    elif "-" in period and len(period) < 8:
        
        # comparison of two GWLs
        if not all([p in runs["EuroCORDEX"].period for p in  period.split("-")]):
            print("Some warming levels do not appear in time slices")
            return
        
        # split period into WLs of interest, select & compute difference
        runs = {k : v.sel(period = period.split("-")).diff("period").squeeze(drop = True) for k, v in runs.items() if not "ERA" in k}
        
        # filter out any runs where there is no data (eg. at 3deg warming, which some runs do not reach)
        runs = {k : ens.sel(run = ~np.isnan(ens.mean(xy))) for k, ens in runs.items()}
        
    else:
        # future periods: return difference between future and baseline periods (excluding evaluation runs)
        # confirm that period actually exists
        if not period in runs["EuroCORDEX"].period:
            print("Invalid period")
            return
        
        # for comparison of return periods there is no baseline period, compute against given RP
        if varnm[-4:] in ["rp20", "rp50"]:
            runs = {k : v.sel(period = period) - int(varnm[-2:]) for k, v in runs.items() if not "ERA" in k}
        else:
            runs = {k : v.sel(period = period) - v.sel(period = "19801201-20101130") for k, v in runs.items() if not "ERA" in k}
            
        # filter out any runs where there is no data (eg. at 3deg warming, which some runs do not reach)
        runs = {k : ens.sel(run = ~np.isnan(ens.mean(xy))) for k, ens in runs.items()}
    
    # optional: remove any extreme values (currently defined as more than 5 times iqr above q99 or, if q01 < 0, less than 5 times iqr below q01)
    if trim:
        zz = xr.concat([v.stack(s = ["run", "projection_y_coordinate", "projection_x_coordinate"]).dropna("s") for k, v in runs.items() if not "obs" in k], "s")
        qq = zz.quantile([0.01, 0.25, 0.75, 0.99])
        iqr = (qq.sel(quantile = 0.75) - qq.sel(quantile = 0.25))
        ul = qq.sel(quantile = 0.99) + 5 * iqr
        ll = qq.sel(quantile = 0.01) - 5 * iqr

        if qq.sel(quantile = 0.01) < 0:
            runs = {k : v.where(v <= ul).where(v >= ll) for k, v in runs.items()}
        else: 
            runs = {k : v.where(v <= ul) for k, v in runs.items()}
            
    # label data with period for easier reference later on
    runs = {k : v.assign_coords(period = ([],period)) for k, v in runs.items()}
            
    # check for no non-zero values
    if all([v.mean().values.tolist() == 0 for v in list(runs.values())]):
        if verbose: print("0", end = "")
        return
    else:
        return runs
    
####################################################################################################################

def plot_title(plottype, varnm, period, season):
    
    if season in ["DJF", "MAM", "JJA", "SON"]:
        sstr = " "+season+" "
    else:
        sstr = ""
    
    if period == "19890101-20081231":
        ctype = "bias in "
        pname ="(" + sstr + "19890101-20081231)"
    elif "-" in period:
        ctype = "change in "
        pname = "(" + sstr + period+")"
    else:
        ctype = "change in "
        if sstr != "": sstr == "("+sstr+")"
        pname = "after GMST increase of "+period+"°C" + sstr
    
    plot_strings = {'Taylor-diagram' : "Taylor diagram of ",
                    'boxplots' : "Boxplots of "+ctype,
                    'ens-mean-maps' : "Ensemble mean "+ctype,
                    'ens-sd-maps' : "Ensemble standard deviation of "+ctype,
                    'epp-gcm' : "GCM EPPs of "+ctype,
                    'epp-rcm' : "RCM EPPs of "+ctype,
                    'epp-ukcp' : "UKCP18 12km EPPs of "+ctype,
                    'eurocordex-anova' : "Decomposition of sources of variance in EuroCORDEX ensemble of "+ctype,
                    'stampplots-cmip5' : "Maps of "+ctype,
                    'stampplots-eurocordex' : "Maps of "+ctype,
                    'stampplots-ukcp' : "Maps of "+ctype}
    
    return plot_strings[plottype]+ ctype + lc(var_cv[varnm]["long_name"]) + " ("+varnm+")"+" "+pname

####################################################################################################################

def ens_mean_maps(varnm = None, period = None, season = None, data = None, save = True, fmin = None, fmax = None, trim = False):
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
    
    # get ensemble means & plotting limits
    ens_means = {k : v.mean("run", skipna = True).reset_coords(drop = True) for k, v in data.items() if not k == "obs"}
    ens_names = list(ens_means.keys())
    vlims = vrange(xr.concat(ens_means.values(), "src"))
    extend_cbar = "neither"
    
    # if max/min plotting limits are given, check if they are exceeded and truncate vlims accordingly
    if fmin is not None and fmax is not None:
        vmin, vmax, extend_cbar = fix_cbar(fmin,fmax,**vlims).values()
        vlims = {"vmin" : vmin, "vmax" : vmax}
    
    # draw the plots
    fig, axs = ukmap_subplots(ncols = len(data), figsize = (len(data)*3 - 2,4))
    
    if period == "19890101-20081231":
        o = 1
        obs_vlims = vrange(data["obs"])
        cbar_obs = qmap(data["obs"], ax = axs[0], colorbar = False, **obs_vlims, cmap = fix_cmap(var_cv[varnm]["cmap"], **obs_vlims),
                        title = "HadUK-grid (observed)\n("+str(data["obs"].mean().values.round(1))+var_cv[varnm]["label_units"]+")")
        plt.colorbar(cbar_obs, ax = axs[0], location = "bottom", fraction = 0.05, pad = 0.01, label = var_cv[varnm]["plot_label"])
        cbar_label = "Bias in "+lc(var_cv[varnm]["plot_label"])
    else:
        o = 0
        if period == "19801201-20101130":
            cbar_label = var_cv[varnm]["plot_label"]
        else:
            cbar_label = "Change in "+lc(var_cv[varnm]["plot_label"])
    
    for i in range(len(ens_means)):
        cbar_ch = qmap(ens_means[ens_names[i]], ax = axs[o+i], colorbar = False, **vlims, cmap = fix_cmap(var_cv[varnm]["cmap"], **vlims),
                      title = ens_names[i] + "\n("+str(ens_means[ens_names[i]].mean().values.round(1))+var_cv[varnm]["label_units"]+")")
        
    plt.colorbar(cbar_ch, ax = axs[o:], location = "bottom", fraction = 0.05, pad = 0.01, label = cbar_label, extend = extend_cbar)
    
    if save:
        plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_ens-mean-maps_"+period+"_"+season+".png"); plt.close()
        if o == 1:
            data_out = xr.concat([data["obs"].reset_coords(drop = True)] + list(ens_means.values()), "ensemble").assign_coords(ensemble = ["obs"] + ens_names).expand_dims(period = [period], season = [season])
        else:
            data_out = xr.concat(list(ens_means.values()), "ensemble").assign_coords(ensemble = ens_names).expand_dims(period = [period], season = [season])
        data_out.to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_ens-mean-maps_"+period+"_"+season+".nc")
        print(".", end = "")
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ens_sd_maps(varnm = None, period = None, season = None, data = None, save = True, trim = False):
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
    
    data = {k : v for k, v in data.items() if not "obs" in k}
    
    # get ensemble means & plotting limits
    ens_sds = {k : v.std("run", skipna = True).reset_coords(drop = True) for k, v in data.items()}
    ens_names = list(data.keys())
    vlims = vrange(xr.concat([v for k, v in ens_sds.items() if not k == "obs"], "src"))

    # draw the plots
    fig, axs = ukmap_subplots(ncols = len(data))
    
    if period == "19890101-20081231":
        cbar_label = "Bias in "+lc(var_cv[varnm]["plot_label"])
    elif period == "19801201-20101130":
        cbar_label = var_cv[varnm]["plot_label"]
    else:
        cbar_label = "Change in "+lc(var_cv[varnm]["plot_label"])
    
    for i in range(len(ens_sds)):
        cbar_ch = qmap(ens_sds[ens_names[i]], ax = axs[i], colorbar = False, title = ens_names[i], **vlims, cmap = fix_cmap(var_cv[varnm]["cmap"], **vlims))
        
    plt.colorbar(cbar_ch, ax = axs, location = "bottom", fraction = 0.05, pad = 0.01, label = cbar_label)
    
    if save:
        plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_ens-sd-maps_"+period+"_"+season+".png"); plt.close()
        data_out = xr.concat(list(ens_sds.values()), "ensemble").assign_coords(ensemble = ens_names).expand_dims(period = [period], season = [season])
        data_out.to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_ens-sd-maps_"+period+"_"+season+".nc")
        print(".", end = "")
    
    
####################################################################################################################

def boxplots(varnm = None, period = None, season = None, data = None, save = True, trim = False):
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
    
    ens_names = [k for k in data.keys() if not k == "obs"]
    
    # set up legend
    bplot_handles = [matplotlib.lines.Line2D([], [], color = "darkred", linestyle = "--"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "-"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "--")]
    bplot_labels = ["HadUK-grid", "Ensemble median", "Ensemble mean"]
    
    # UK averages
    if period == "19890101-20081231":
        uk_means = {k : (v + data["obs"].where(~np.isinf(data["obs"])).squeeze(drop = True)).mean(xy, skipna = True).dropna("run").reset_coords(drop = True) for k, v in data.items() if not k == "obs"}
        hline = data["obs"].mean(xy, skipna = True).values.tolist()
        ylabel = var_cv[varnm]["plot_label"]
    else:
        uk_means = {k : v.mean(xy, skipna = True).dropna("run").reset_coords(drop = True) for k, v in data.items()}
        ylabel = "Change in "+lc(var_cv[varnm]["plot_label"])
        hline = 0
        bplot_handles = bplot_handles[1:]
        bplot_labels = bplot_labels[1:]
        
    if period == "19801201-20101130":
        ylabel = var_cv[varnm]["plot_label"]
        hline = np.nan

    uk_means = xr.concat([v.assign_coords(ensemble = ("run", np.repeat(k, len(v.run)))) for k, v in uk_means.items()], "run").assign_attrs(hline = hline)
    
    # draw the boxplots
    fig, ax = plt.subplots(1,1, figsize = (10,5), dpi= 100, facecolor='w', edgecolor='k')
    
    ax.violinplot([uk_means.sel(run = uk_means.ensemble == ens) for ens in ens_names], showextrema = False)
    ax.boxplot([uk_means.sel(run = uk_means.ensemble == ens) for ens in ens_names], labels = [re.sub(" ", "\n", re.sub("ERA-","ERA-\n",n)) for n in ens_names if not n == "obs"],
               meanline = True, showmeans = True, zorder = 3, meanprops = {"color" : "black"}, medianprops = {"color" : "black"}, flierprops = {"marker" : ""}, widths = 0.5)
    
    # add points
    j = 0.025
    offsets = dict(zip([gcm_full_names[r] for r in cmip5_ec], list(np.arange(j*-4.5, j*5.5, j).round(3))))
    
    if "CMIP5-13" in ens_names:
        i = ens_names.index("CMIP5-13")
        ens_means = uk_means.sel(run = uk_means.ensemble == "CMIP5-13")
        mscatter([offsets[gcm_full_names[n]]+i+1 if n in list(gcm_full_names.keys()) else 0+i+1 for n in ens_means.run.values], ens_means,
                 m = [gcm_markers[mdl] for mdl in ens_means.run.values], color = gcm_colours(ens_means.run.values), edgecolor = "k", s = 50, zorder = 5, ax = ax)
        
    if "CMIP5-EC" in ens_names:
        i = ens_names.index("CMIP5-EC")
        ens_means = uk_means.sel(run = uk_means.ensemble == "CMIP5-EC")
        mscatter([offsets[gcm_full_names[n]]+i+1 for n in ens_means.run.values], ens_means,
                 m = [gcm_markers[mdl] for mdl in ens_means.run.values], color = gcm_colours(ens_means.run.values), edgecolor = "k", s = 50, zorder = 5, ax = ax)
        
    if "EuroCORDEX" in ens_names:
        i = ens_names.index("EuroCORDEX")
        ens_means = uk_means.sel(run = uk_means.ensemble == "EuroCORDEX")
        rcm_gcms = [re.sub("p1_.+", "p1", rn) for rn in ens_means.run.values]
        rcm_rcms = [re.sub(".+_", "", rn) for rn in ens_means.run.values]
        mscatter([offsets[n]+i+1 for n in rcm_gcms], ens_means,
                 color = [rcm_colours[n] for n in rcm_rcms], m = [gcm_markers[n] for n in rcm_gcms], edgecolor = "k", zorder = 5, s = 50, ax = ax)
        
    if "ERA-EuroCORDEX" in ens_names:
        i = ens_names.index("ERA-EuroCORDEX")
        ens_means = uk_means.sel(run = uk_means.ensemble == "ERA-EuroCORDEX")
        eval_rcms = [re.sub(".+_", "", rn) for rn in ens_means.run.values]
        ax.scatter(np.repeat(i+1, len(ens_means)), ens_means, 
                   color = [rcm_colours[n] for n in eval_rcms], marker = gcm_markers["ECMWF-ERAINT_r1i1p1"], edgecolor = "k", zorder = 5, s = 50)
        
    if "UKCP18 60km" in ens_names:
        i = ens_names.index("UKCP18 60km")
        ens_means = uk_means.sel(run = uk_means.ensemble == "UKCP18 60km")
        mscatter(np.repeat(i+1, len(ens_means)), ens_means, color = "black", m = "o", edgecolor = "k", zorder = 5, s = 50, ax = ax)
        mscatter(i+1, ens_means[0], color = rcm_colours["HadREM3-GA7-05"], marker = "o", edgecolor = "k", zorder = 5, s = 50, ax = ax)
        
    if "UKCP18 12km" in ens_names:
        i = ens_names.index("UKCP18 12km")
        ens_means = uk_means.sel(run = uk_means.ensemble == "UKCP18 12km")
        mscatter(np.repeat(i+1, len(ens_means)), ens_means, color = "black", m = "o", edgecolor = "k", zorder = 5, s = 50, ax = ax)
        mscatter(i+1, ens_means[0], color = rcm_colours["HadREM3-GA7-05"], marker = "o", edgecolor = "k", zorder = 5, s = 50, ax = ax)
    
    # add gridlines, axis label, legend
    plt.grid(axis = "y", ls = "--")
    ax.axhline(hline, color = "darkred", linestyle = "--")
    ax.set_ylabel(ylabel)
    ax.legend(handles = bplot_handles, labels = bplot_labels, edgecolor = "white", loc = "best")
    
    
    # add model legend
    ms = 7
    fs = 0.92
    gcm_legend(gcm_names = [gcm for gcm in cmip5_13 if not gcm in cmip5_ec], loc='upper left', bbox_to_anchor = (fs, 0.9), title = "CMIP5-13 GCMs", ncol = 2, markersize = ms)
    gcm_legend(gcm_names = cmip5_ec, loc='upper left', bbox_to_anchor = (fs, 0.61), title = "CMIP5-EC GCMs", ncol = 2, markersize = ms)
    rcm_legend(loc='upper left', bbox_to_anchor = (fs, 0.32), title = "EuroCORDEX RCMs", ncol = 2, markersize = ms)

    if save:
        plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_boxplots_"+period+"_"+season+".png"); plt.close()
        uk_means.expand_dims(period = [period], season = [season]).to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_boxplots_"+period+"_"+season+".nc")
        print(".", end = "")

        
####################################################################################################################

def model2_epps(theta_hat, da, n_pcs = 2):
    
    if "season" in da.dims:
        ybar = da.isel(season = 0).mean("run", skipna = False)
    else:
        ybar = da.mean("run", skipna = False)
    
    # principal component analysis of fitted alpha & beta produced by fit_model2
    epps = []
    for theta in [theta_hat["alpha_hat"], theta_hat["beta_hat"]]:
        u, s, vh = np.linalg.svd(theta, full_matrices = False)
        tss = sum(np.diag(vh.transpose() @ np.diag(s**2) @ vh))
        svecs = xr.concat([vec2map(theta_hat["mu_hat"], ybar)] + [vec2map(vh[n,:], ybar) for n in range(n_pcs)], dim = "pc")
               
        ds = xr.Dataset(data_vars = { "svecs"    : (svecs.dims, svecs.data),
                                      "svals"    : (["pc"], np.append(np.array(1), s[:n_pcs])),
                                      "var_expl" : (["pc"], np.append(np.array(1), ((s**2) / tss * 100)[:n_pcs])),
                                      "scores"   : (["run", "pc"], np.column_stack([np.ones(u[:,1].shape), u[:,:n_pcs]]))},
                        coords = svecs.coords).assign_coords(pc = ["mean"] + list(range(n_pcs+1)[1:]))
    
        # correct signs of PCs so that positive score denotes an increase over the UK
        sc_adj = np.sign(svecs.mean(["projection_x_coordinate", "projection_y_coordinate"]))
        sc_adj[0] = 1
        ds["svecs"] = ds["svecs"] * sc_adj
        ds["scores"] = ds["scores"] * sc_adj
        
        epps.append(ds)       
    
    epps[0] = epps[0].assign_coords(run = theta_hat["gcms"])
    epps[1] = epps[1].assign_coords(run = theta_hat["rcms"])    
    
    return epps

    
def epp_plots(varnm = None, period = None, season = None, data = None, save = True, trim = False):
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
    
    cmap = var_cv[varnm]["cmap"]
    cbar_label = var_cv[varnm]["plot_label"]
    
    if period == "19890101-20081231":
        data["EuroCORDEX"] = xr.concat([data["EuroCORDEX"], data["ERA-EuroCORDEX"]], "run")
        ctype = " bias"
    elif period == "19801201-20101130":
        ctype = ""
    else:
        cbar_label = "Change in "+lc(cbar_label)
        ctype = " change"
        
    # fit EPPs for GCM / RCM / UKCP
    theta_hat = fit_model2(data["EuroCORDEX"])
    gcm, rcm = model2_epps(theta_hat, data["EuroCORDEX"])
    ukcp = qpca(data["UKCP18 12km"], n_pcs = 2)
        
    # produce plots & save
    for ens_nm, ens in zip(["gcm", "rcm", "ukcp"], [gcm, rcm, ukcp]):
        
        pca_plot(ens, cmap = cmap, markers = None, colours = None, fs = 12, cbar_label = cbar_label)
        (plt.gcf()).axes[0].set_title("Ensemble mean"+ctype, size = 12)
        
        if save:
            plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_epp-"+ens_nm+"_"+period+"_"+season+".png"); plt.close()
        
            ens["epps"] = ens.svecs * ens.svals
            ens = ens.drop(["svecs", "svals"]).expand_dims(period = [period], season = [season]).assign_attrs(varnm = varnm)
            ens.to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_epp-"+ens_nm+"_"+period+"_"+season+".nc")
    
    print(".", end = "")
        
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cordex_anova(varnm = None, period = None, season = None, data = None, save = True, trim = False):
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
    
    if period == "19890101-20081231":
        cbar_label = var_cv[varnm]["plot_label"]
    elif period == "19801201-20101130":
        cbar_label = var_cv[varnm]["plot_label"]
    else:
        cbar_label = "Change in "+lc(var_cv[varnm]["plot_label"])
        
    obs_ve, est_ve, est_ve_maps = unbalanced_manova(data["EuroCORDEX"]).values()
           
    def ve_str(v): return "\n("+str(int((est_ve[v]).round(0)))+"%)"
    
    fig, axs = ukmap_subplots(ncols = 4, figsize = (15,5))
    cbar_ss = qmap(est_ve_maps.sel(src = "sd"), ax = axs[0], colorbar = False, cmap = fix_cmap(var_cv[varnm]["cmap"], vmin = 1, vmax = 1), title = "Ensemble standard deviation\n")
    cbar_ve = qmap(est_ve_maps.sel(src = "gcm"), ax = axs[1], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "Proportion explained by GCM"+ve_str("estve_G2"))
    qmap(est_ve_maps.sel(src = "rcm"), ax = axs[2], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "Proportion explained by RCM"+ve_str("estve_R2"))
    qmap(est_ve_maps.sel(src = "res"), ax = axs[3], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "Residual uncertainty"+ve_str("estve_res"))
    
    plt.colorbar(cbar_ss, ax = axs[0], location = "bottom", pad = 0.02, fraction = 0.05, label = cbar_label)
    plt.colorbar(cbar_ve, ax = axs[1:], location = "bottom", pad = 0.02, fraction = 0.03, label = "% of variance explained by each component")
    
    if save:
        
        plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_eurocordex-anova_"+period+"_"+season+".png"); plt.close()
        
        # add proportions of variance explained, expand period & season to allow direct plotting
        est_ve_maps = est_ve_maps.assign_attrs(obs_ve).assign_attrs(est_ve).expand_dims(period = [period], season = [season])
        est_ve_maps.to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_eurocordex-anova_"+period+"_"+season+".nc")
        print(".", end = "")
     
    
####################################################################################################################

def stampplots(varnm = None, period = None, season = None, data = None, save = True, trim = False, cmip = True, ukcp = True, ec = True):
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
            
    if period == "19890101-20081231":
        cbar_label = "Bias in "+lc(var_cv[varnm]["plot_label"])
        ec_data = xr.concat([data["EuroCORDEX"], data["ERA-EuroCORDEX"]], "run")
        ec_gcms = ["ECMWF-ERAINT_r1i1p1"] + [v for v in gcm_full_names.values() if not "ECMWF" in v]
        ecmwf = 1
        fig_width = 26
    else:
        cbar_label = "Change in "+lc(var_cv[varnm]["plot_label"])
        ec_data = data["EuroCORDEX"]
        ec_gcms = [v for v in gcm_full_names.values() if not "ECMWF" in v]
        ecmwf = 0
        fig_width = 23
    if period == "19801201-20101130":
        cbar_label = var_cv[varnm]["plot_label"]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CMIP stamp plots
    
    if cmip:
        if "CMIP5-13" in list(data.keys()) and "CMIP5-EC" in list(data.keys()):
            cmip_data = xr.concat([data["CMIP5-13"], data["CMIP5-EC"]], "run").drop_duplicates("run")
            cmip_vlims = vrange(cmip_data)
            cmip_cmap = fix_cmap(var_cv[varnm]["cmap"], **cmip_vlims)
            
            # Assuming no GCM data exists for EC-EARTH r3i1p1: will need to rejig if it becomes available
            fig, axs = ukmap_subplots(ncols = 9, nrows = 2, figsize = (16,6))
            fig.subplots_adjust(hspace = 0.25, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
            
            for gcm in cmip_data.run.values:
                if gcm in cmip5_ec:
                    i = [nm for nm in cmip5_ec if not nm == "EC-EARTH_r3i1p1"].index(gcm)
                else:
                    i = [nm for nm in cmip5_13 if not nm in cmip5_ec].index(gcm) + 9
                ax = fig.axes[i]
                cbar = qmap(cmip_data.sel(run = gcm), ax = ax, cmap = cmip_cmap, **cmip_vlims, colorbar = False)
                ax.set_title(re.sub("_", "\n", gcm))            
            axs[0,0].text(-0.07, 0.55, "CMIP5-EC", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes)
            axs[1,0].text(-0.07, 0.55, "CMIP5-13", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[1,0].transAxes)
            
            plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.03, pad = 0.01, label = cbar_label)
            
            if save:
                plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_stampplots-cmip5_"+period+"_"+season+".png"); plt.close()
                cmip_data.expand_dims(period = [period], season = [season]).to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_stampplots-cmip5_"+period+"_"+season+".nc")
                print(".", end = "")
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # UKCP18 stamp plots
    
    if ukcp:
        if "UKCP18 60km" in list(data.keys()):
            ukcp_data = xr.concat([data["UKCP18 12km"], data["UKCP18 60km"]], "resolution").assign_coords(resolution = ["12km", "60km"])
            nr = 2
        else:
            ukcp_data = data["UKCP18 12km"].expand_dims(resolution = ["12km"])
            nr = 1
        ukcp_vlims = vrange(ukcp_data)
        ukcp_cmap = fix_cmap(var_cv[varnm]["cmap"], **ukcp_vlims)
        
        fig, axs = ukmap_subplots(ncols = 12, nrows = nr, figsize = (20,3 * nr))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if "UKCP18 60km" in list(data.keys()):
            for i in range(12):
                cbar = qmap(ukcp_data.isel(run = i, resolution = 0), ax = axs[1,i], cmap = ukcp_cmap, **ukcp_vlims, colorbar = False, title = data["UKCP18 12km"].run.values.tolist()[i])
                qmap(ukcp_data.isel(run = i, resolution = 1), ax = axs[0,i], cmap = ukcp_cmap, **ukcp_vlims, colorbar = False)
                
            axs[0,0].text(-0.07, 0.55, "60km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
            axs[1,0].text(-0.07, 0.55, "12km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[1,0].transAxes, fontsize = "large")  
        else:
            for i in range(12):
                cbar = qmap(ukcp_data.isel(run = i, resolution = 0), ax = axs[i], cmap = ukcp_cmap, **ukcp_vlims, colorbar = False, title = data["UKCP18 12km"].run.values.tolist()[i])
                axs[0].text(-0.07, 0.55, "12km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0].transAxes, fontsize = "large")
                
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.03, pad = 0.01, label = cbar_label)
    
        if save:
            plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_stampplots-ukcp_"+period+"_"+season+".png"); plt.close()
            ukcp_data.expand_dims(period = [period], season = [season]).to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_stampplots-ukcp_"+period+"_"+season+".nc")
            print(".", end = "")
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # EuroCORDEX stamp plots
    
    if ec:
        if "CMIP5-EC" in list(data.keys()):
            ec_cmip_data = xr.concat([ec_data.assign_coords(ensemble = ("run", np.repeat("EuroCORDEX", len(ec_data.run)))),
                                  data["CMIP5-EC"].assign_coords(ensemble = ("run", np.repeat("CMIP5", len(data["CMIP5-EC"].run))))], "run")
            ec_vlims = vrange(xr.concat([ec_data, data["CMIP5-EC"]], "run"))
            cmip = 1
        else:
            ec_cmip_data = ec_data.assign_coords(ensemble = ("run", np.repeat("EuroCORDEX", len(ec_data.run))))
            ec_vlims = vrange(ec_data)
            cmip = 0
            
        ec_cmap = fix_cmap(var_cv[varnm]["cmap"], **ec_vlims)
        
        fig, axs = ukmap_subplots(ncols = 10 + ecmwf, nrows = 10 + cmip, figsize = (fig_width, 24))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        # plots of RCM runs
        for rcm_nm in ec_data.run.values:
            rcm_i = list(rcm_colours.keys()).index(re.sub(".+_", "", rcm_nm))
            gcm_i = ec_gcms.index(re.sub("p1_.+","p1",rcm_nm))
            cbar = qmap(ec_data.sel(run = rcm_nm), ax = axs[rcm_i++cmip, gcm_i], cmap = ec_cmap, **ec_vlims, colorbar = False)
            
        if "CMIP5-EC" in list(data.keys()):
            
            for gcm_nm in data["CMIP5-EC"].run.values:
                gcm_i = [k for k in gcm_full_names.keys() if not "ERAINT" in k].index(gcm_nm) + ecmwf
                qmap(data["CMIP5-EC"].sel(run = gcm_nm), ax = axs[0,gcm_i], cmap = ec_cmap, **ec_vlims, colorbar = False)
                
            axs[0,0].text(-0.07, 0.55, "GCM", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
            
        for i in range(10+ecmwf): axs[0,i].set_title(re.sub("_", "\n", {v : k for k, v in gcm_full_names.items()}[ec_gcms[i]]))
        for j in range(10): axs[j+cmip,0].text(-0.07, 0.55, list(rcm_colours.keys())[j], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[j+cmip,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.01, pad = 0.01, label = cbar_label)
        
        
        if save:
            plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_stampplots-eurocordex_"+period+"_"+season+".png"); plt.close()
            ec_cmip_data.expand_dims(period = [period], season = [season]).to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_stampplots-eurocordex_"+period+"_"+season+".nc")
            print(".", end = "")
        

####################################################################################################################

def Taylor_diagram(varnm = None, period = None, season = None, data = None, ref = None, save = True, trim = False):
    
    if period == "19801201-20101130": 
        print("~", end = "")
        return
    
    if data is not None:
        varnm = data["EuroCORDEX"].name
        period = data["EuroCORDEX"].period.values.tolist()
        season = data["EuroCORDEX"].season.values.tolist()
    else:
        # use data provided, otherwise load data
        data = load_data(varnm, period, season, trim = trim)
        if data is None: return
    
    if period == "19890101-20081231":
        corrs = {k : xr.corr(data[k] + data["obs"].squeeze(drop = True), data["obs"].squeeze(drop = True), xy) for k in data.keys() if not "obs" in k}
        sds = {k : (data[k] + data["obs"].squeeze(drop = True)).std(xy) / data["obs"].std(xy).values for k in data.keys() if not "obs" in k}
        ref_sd = data["obs"].std(xy).values.tolist()
    else:
        # for future periods, need to reconstruct projected field & compare to baseline
        if ref is None:
            ref = load_data(varnm, "19801201-20101130", season)
        corrs = {k : xr.corr(data[k] + ref[k], ref[k], xy) for k in data.keys()}
        sds = {k : (data[k] + ref[k]).std(xy) / ref[k].std(xy) for k in data.keys()}
        ref_sd = 1
        
    # remove any runs with NA correlations
    corrs = {k : v.dropna("run", "any") for k, v in corrs.items()}
    sds = {k : v.sel(run = np.isin(v.run, corrs[k].run)) for k, v in sds.items()}
        
    # compile everything into a single dataset to make saving easier
    tdata = xr.Dataset(data_vars = { "corrs" : xr.concat([v.assign_coords(ensemble = ("run", np.repeat(k, len(v.run)))) for k, v in corrs.items()], "run"),
                                     "sds" : xr.concat([v.assign_coords(ensemble = ("run", np.repeat(k, len(v.run)))) for k, v in sds.items()], "run")}).assign_attrs(ref_sd = ref_sd)
        
    max_sd = min(np.ceil(max(tdata.sds).values.tolist()) / ref_sd, 5)
    min_corr = min(tdata.corrs).values.tolist()
    
    if min_corr < 0:
        w = 12; extend = True
    else:
        w = 6; extend = False
        
    fig = plt.figure(figsize = (w,6), dpi= 100, facecolor='w', edgecolor='k')
    
    td = TaylorDiagram(tdata.ref_sd, fig = fig, label = "Baseline", srange = (0,max_sd), extend = extend)
    td.add_contours(colors = "mistyrose")
    td.add_grid(color = "papayawhip")
     
    if "CMIP5-13" in tdata.ensemble:
        ens = tdata.sel(run = tdata.ensemble == "CMIP5-13")
        for r in ens.run.values:
            td.add_sample(ens.sds.sel(run = r).values, ens.corrs.sel(run = r).values, marker = gcm_markers[r], label = "_", ms = 7, ls = '', mfc = gcm_colours([r]), mec = "black")
    
    if "CMIP5-EC" in tdata.ensemble:
        ens = tdata.sel(run = tdata.ensemble == "CMIP5-EC")
        for r in ens.run.values:
            td.add_sample(ens.sds.sel(run = r).values, ens.corrs.sel(run = r).values, marker = gcm_markers[r], label = "_", ms = 7, ls = '', mfc = gcm_colours([r]), mec = "black")
    
    if "EuroCORDEX" in tdata.ensemble:
        ens = tdata.sel(run = tdata.ensemble == "EuroCORDEX")
        for r in ens.run.values:
            td.add_sample(ens.sds.sel(run = r).values, ens.corrs.sel(run = r).values, marker = gcm_markers[re.sub("p1_.+", "p1", r)], label = "_", ms = 7, ls = '', mfc = rcm_colours[re.sub(".+_", "", r)], mec="black")
    
    if "ERA-EuroCORDEX" in tdata.ensemble:
        ens = tdata.sel(run = tdata.ensemble == "ERA-EuroCORDEX")
        for r in ens.run.values:
            td.add_sample(ens.sds.sel(run = r).values, ens.corrs.sel(run = r).values, marker = gcm_markers[re.sub("p1_.+", "p1", r)], label = "_", ms = 7, ls = '', mfc = rcm_colours[re.sub(".+_", "", r)], mec="black")
    
    if "UKCP18 60km" in tdata.ensemble:
        ens = tdata.sel(run = tdata.ensemble == "UKCP18 60km")
        for r in ens.run.values:
            if r == 1:
                mcolour = rcm_colours["HadREM3-GA7-05"]
            else:
                mcolour = "black"
        td.add_sample(ens.sds.sel(run = r).values, ens.corrs.sel(run = r).values,  marker = "x", label = "_", ms = 7, ls = '', mfc = mcolour, mec = mcolour)
    
    if "UKCP18 12km" in tdata.ensemble:
        ens = tdata.sel(run = tdata.ensemble == "UKCP18 12km")
        for r in ens.run.values:
            if r == 1:
                mcolour = rcm_colours["HadREM3-GA7-05"]
            else:
                    mcolour = "black"
            td.add_sample(ens.sds.sel(run = r).values, ens.corrs.sel(run = r).values, marker = "+", label = "_", ms = 7, ls = '', mfc = mcolour, mec = mcolour)
    
    ms = 7
    fs = 0.95
    rcm_legend(loc='upper left', bbox_to_anchor = (fs, 0.95), title = "EuroCORDEX RCMs", ncol = 2, markersize = ms)
    gcm_legend(gcm_names = cmip5_ec, loc='upper left', bbox_to_anchor = (fs, 0.7), title = "CMIP5-EC GCMs", ncol = 2, markersize = ms)
    gcm_legend(gcm_names = [gcm for gcm in cmip5_13 if not gcm in cmip5_ec], loc='upper left', bbox_to_anchor = (fs, 0.45), title = "CMIP5-13 GCMs", ncol = 2, markersize = ms)
    gcm_legend(gcm_names = ["UKCP18 12km", "UKCP18 60km", "ERAINT_r1i1p1", "HadUK-Grid"], loc='upper left', bbox_to_anchor = (fs, 0.2), title = "Other runs", ncol = 2, markersize = ms)
 
    if save:
        plt.savefig("/data/met/plot-explorer/images/"+varnm+"/"+varnm+"_12km_Taylor-diagram_"+period+"_"+season+".png"); plt.close()
        tdata.expand_dims(period = [period], season = [season]).to_netcdf("/data/met/plot-explorer/data/"+varnm+"/"+varnm+"_12km_Taylor-diagram_"+period+"_"+season+".nc")
        print(".", end = "")