
import xarray as xr
import pandas as pd
import glob
import re

import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/met/processing/10_methods')
from taylorDiagram import TaylorDiagram
from misc import load_eval, load_diff, xy
from dicts import *
from plotting import *
from pc_decomposition import qpca, qpca_plot

# default to tight bounding box
matplotlib.rcParams['savefig.bbox'] = "tight"

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore", message = "FixedFormatter should only be used together with FixedLocator")  # something about adding vertical gridlines to variance explained


####################################################################################################################

def change_maps(varnm, season, period = "20491201-20791130", rel_diff = False, recover_clim = False, cmap = None, contour = False, contour_interval = 1, contour_offset = False, save = False, title = False, figsize = None, 
                show_mean = True, mean_dp = 1, cbar_loc = "right", vmin = None, vmax = None, remove_13 = False):
    
    # load the data
    data = load_diff(varnm = varnm, season = season, period = period, rel_diff = rel_diff, recover_clim = recover_clim)
    if remove_13: data = {re.sub("CMIP5-EC", "CMIP5", k):v for k, v in data.items() if not k == "CMIP5-13"}
        
    ens_list = list(data.values())
    ens_names = list(data.keys())
    
    c_units = var_cv[varnm]["label_units"]
    
    if rel_diff:
        main = "Relative projected change in "+season+" " + lc(var_cv[varnm]["plot_label"])+" over the UK between 1980-2010 and "+period[:4]+"-"+period[9:13]
        cbar_label = "% change in " + re.sub("\(.+\)","",lc(var_cv[varnm]["plot_label"]))
        c_units = "%"
        d = ""
    elif recover_clim:
        main = "Average "+season+" "+varnm+" over the UK from "+period[:4]+"-"+period[9:13]
        cbar_label = var_cv[varnm]["plot_label"]
        d = ""
    else:
        main = "Projected change in "+season+" " + lc(var_cv[varnm]["plot_label"])+" over the UK between 1980-2010 and "+period[:4]+"-"+period[9:13]
        cbar_label = "Change in " + re.sub("air", "\nair", lc(var_cv[varnm]["plot_label"]))
        d = ""
        
    ens_change = [ens.mean("run", skipna = False).reset_coords(drop = True) for ens in ens_list]
    
    vlims = vrange(xr.concat(ens_change, "src"))
    if vmin is None: vmin = vlims["vmin"]
    if vmax is None: vmax = vlims["vmax"]

    if show_mean:
        means = [ec.mean().values.round(mean_dp) for ec in ens_change]
        if mean_dp == 0: means = [int(m) for m in means]
        means = ["\n("+str(m)+c_units+")" for m in means]
    else:
        means = ["" for ec in ens_change]
        
    if not figsize: figsize = (len(data)*3 - 2,4)
    if not cmap: cmap = var_cv[varnm]["cmap"]

    fig, axs = ukmap_subplots(ncols = len(data), figsize = figsize)
    if title:
        fig.suptitle(main, fontweight = "bold")
       
    for i in range(len(ens_change)):
        cbar_ch = qmap(ens_change[i], ax = axs[i], colorbar = False, title = ens_names[i]+ means[i], vmin = vmin, vmax = vmax, cmap = cmap, 
                       contour = contour, contour_interval = contour_interval, contour_offset = contour_offset)
    
    if cbar_loc == "right":
        plt.colorbar(cbar_ch, ax = axs, location = "right", fraction = 0.05, pad = 0.01, shrink = 0.95, label = cbar_label)
    else:
        plt.colorbar(cbar_ch, ax = axs, location = "bottom", fraction = 0.05, pad = 0.01, label = cbar_label)
    
    if save: 
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_change-maps-relative.png", bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_change-maps.png")
        
        
#=================================================================================================================

def change_boxplots(varnm, season, period = "20491201-20791130", recover_clim = False, rel_diff = False, model_legend = False, legend_w = 0.65, save = False, title = False, j = 0.025, ax = None, biaslines = "auto",
                    blwidth = 1, hline = 0, legend_pos = "best", notch = False, bootstrap = None, remove_13 = False):
    
    # load the data
    data = load_diff(varnm, season, period, recover_clim)
    if remove_13: data = {re.sub("CMIP5-EC", "CMIP5", k):v for k, v in data.items() if not k == "CMIP5-13"}
    
    ens_list = list(data.values())
    ens_names = list(data.keys())  
    
    xy = ["projection_x_coordinate", "projection_y_coordinate"]
    if rel_diff:
        # relative differences are reported here as % change over the whole of the UK - not the average of the % changes in each grid cell
        baseline = load_diff(varnm, season, "19801201-20101130")
        if remove_13:
            baseline = {re.sub("CMIP5-EC", "CMIP5", k):v for k, v in baseline.items() if not k == "CMIP5-13"}
        baseline = list(baseline.values())
        means = [ch.mean(xy) / bl.mean(xy) * 100 for ch, bl in zip(ens_list, baseline)]
        ylabel = "% change in " + re.sub(" \(.+\)", "", lc(var_cv[varnm]["plot_label"]))
    else:
        means = [ch.mean(xy) for ch in ens_list]
        ylabel = "Change in " + lc(var_cv[varnm]["plot_label"])
      
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if ax is None:
        if model_legend:
            fig, ax = plt.subplots(1,1, figsize = (12,6), dpi= 100, facecolor='w', edgecolor='k')
        else:
            fig, ax = plt.subplots(1,1, figsize = (8,6), dpi= 100, facecolor='w', edgecolor='k')
    if title: fig.suptitle("Boxplots of mean "+b+season+" "+varnm+" over the UK for each model", fontweight = "bold")
    
    ax.violinplot(means, showextrema = False)
    ax.boxplot(means, labels = [re.sub("ERA-EuroCORDEX","ERA-RCM",n) for n in ens_names], meanline = True, showmeans = True, zorder = 3, meanprops = {"color" : "black"}, medianprops = {"color" : "black"}, flierprops = {"marker" : ""},
               notch = notch, bootstrap = bootstrap)
    
    offsets = dict(zip([gcm_full_names[r] for r in cmip5_ec], list(np.arange(j*-4.5, j*5.5, j).round(3))))
    
    if "CMIP5-13" in ens_names:
        i = ens_names.index("CMIP5-13")
        mscatter([offsets[gcm_full_names[n]]+i+1 if n in list(gcm_full_names.keys()) else 0+i+1 for n in means[i].run.values], means[i],
                 m = [gcm_markers[mdl] for mdl in means[i].run.values], color = gcm_colours(means[i].run.values), edgecolor = "k", s = 50, zorder = 5, ax = ax)
    
    if "CMIP5-EC" in ens_names:
        i = ens_names.index("CMIP5-EC")
        mscatter([offsets[gcm_full_names[n]]+i+1 for n in means[i].run.values], means[i],
                 m = [gcm_markers[mdl] for mdl in means[i].run.values], color = gcm_colours(means[i].run.values), edgecolor = "k", s = 50, zorder = 5, ax = ax)
    if "CMIP5" in ens_names:
        i = ens_names.index("CMIP5")
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
    
    # y-axis label. No titles because these will be added in the main document.
    ax.set_ylabel(ylabel)
    
    if model_legend:    
        # legend for RCM colours & GCM marker styles
        plt.subplots_adjust(right = legend_w)
        rcm_legend(set(rcm_rcms), ax = plt.gcf(), loc = 'upper left', bbox_to_anchor = (legend_w, 0.9))
        gcm_legend(list(gcm_markers.keys())[1:11], ax = plt.gcf(), loc = 'lower left', bbox_to_anchor = (legend_w, 0.1))
    
    # legend for boxplots
    bplot_handles = [matplotlib.lines.Line2D([], [], color = "black", linestyle = "-"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "--")]
    bplot_labels = ["Group median", "Group mean"]
    ax.legend(handles = bplot_handles, labels = bplot_labels, edgecolor = "white", loc = legend_pos)
    
    if save: 
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_boxplots-relative.png", bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_boxplots.png", bbox_inches='tight')
            

#=================================================================================================================

def change_Taylor(varnm, season, period = "20491201-20791130", save = False, legend = False, title = False, max_sd = None, rect = 111, fig = None, extend = False):
    
    # load the data - use climatology fields rather than changes
    data = load_diff(varnm, season, recover_clim = True)
    ens_list = list(data.values())
    ens_names = list(data.keys())
    
    xy = ["projection_y_coordinate", "projection_x_coordinate"]
    
    # compute correlations & relative SDs to get range of axes
    corrs = [xr.corr(ens.sel(period = period), ens.sel(period = "19801201-20101130"), xy) for ens in ens_list]
    rel_sds = [ens.sel(period = period).std(xy).values / ens.sel(period = "19801201-20101130").std(xy) for ens in ens_list]
    sd_range = max(np.concatenate(rel_sds))*1.1
    if max_sd == None:
        max_sd = np.ceil(max(np.concatenate(rel_sds)))
    
    if extend:
        # only extend to accommodate negative correlations if true
        if min(np.concatenate(corrs)) < 0: 
            extend = True
            w = 14
        else: 
            extend = False
            w = 7
    else: w = 7
    
    if fig is None:
        fig = plt.figure(figsize = (w,7), dpi= 100, facecolor='w', edgecolor='k')
        
    td = TaylorDiagram(1, fig = fig, label = "Baseline", srange = (0,max_sd), extend = extend, rect = rect)
    td.add_contours(colors = "mistyrose")
    td.add_grid(color = "papayawhip")
    
    if title:
        plt.suptitle("Taylor diagram evaluating changes in spatial patterns of "+season+" "+varnm+" from "+period+" with respect to baseline period (19801201-20101130)", fontweight = "bold")
    
    if legend:
        rcm_legend(loc='upper left', bbox_to_anchor = (0.95, 0.95))
        leg = gcm_legend(loc='lower left', bbox_to_anchor = (0.95, 0.1))
    
    if "CMIP5-13" in ens_names:
        for r in ens_list[ens_names.index("CMIP5-13")].run.values:
            td.add_sample(rel_sds[ens_names.index("CMIP5-13")].sel(run = r).values, corrs[ens_names.index("CMIP5-13")].sel(run = r).values,
                          marker = gcm_markers[r], label = "_", ms = 7, ls = '', mfc = gcm_colours([r]), mec = "black")
    
    if "CMIP5-EC" in ens_names:
        for r in ens_list[ens_names.index("CMIP5-EC")].run.values:
            td.add_sample(rel_sds[ens_names.index("CMIP5-EC")].sel(run = r).values, corrs[ens_names.index("CMIP5-EC")].sel(run = r).values,
                          marker = gcm_markers[r], label = "_", ms = 7, ls = '', mfc = gcm_colours([r]), mec = "black")
    
    if "EuroCORDEX" in ens_names:
        for r in ens_list[ens_names.index("EuroCORDEX")].run.values:
            td.add_sample(rel_sds[ens_names.index("EuroCORDEX")].sel(run = r).values, corrs[ens_names.index("EuroCORDEX")].sel(run = r).values,
                          marker = gcm_markers[re.sub("p1_.+", "p1", r)], label = "_", ms = 7, ls = '', mfc = rcm_colours[re.sub(".+_", "", r)], mec="black")
    
    if "UKCP18 60km" in ens_names:
        for r in ens_list[ens_names.index("UKCP18 60km")].run.values:
            if r == 1:
                mcolour = rcm_colours["HadREM3-GA7-05"]
            else:
                mcolour = "black"
            td.add_sample(rel_sds[ens_names.index("UKCP18 60km")].sel(run = r).values, corrs[ens_names.index("UKCP18 60km")].sel(run = r).values,
                          marker = "x", label = "_", ms = 7, ls = '', mfc = mcolour, mec = mcolour)
            
    if "UKCP18 12km" in ens_names:
        for r in ens_list[ens_names.index("UKCP18 12km")].run.values:
            if r == 1:
                mcolour = rcm_colours["HadREM3-GA7-05"]
            else:
                mcolour = "black"
            td.add_sample(rel_sds[ens_names.index("UKCP18 12km")].sel(run = r).values, corrs[ens_names.index("UKCP18 12km")].sel(run = r).values, 
                          marker = "+", label = "_", ms = 7, ls = '', mfc = mcolour, mec = mcolour)

    if save: 
        if legend:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_change-Taylor-diagram.png", bbox_extra_artists=(leg), bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_change-Taylor-diagram.png", bbox_inches='tight')
            
            
#=================================================================================================================

def change_pca(varnm, season, period = "20491201-20791130", mtype = "gcm", cmap = None, contour = False, contour_interval = 1, contour_offset = True, save = False, title = False, rel_diff = False, fs = 12):
    
    if rel_diff:
        pca = xr.open_dataset("/data/met/processing/80_pca-anova/change-pca/"+varnm+"_change-pca_"+mtype+"-relative.nc").sel(season = season, period = period)
        cbar_label = "Relative change in "+lc(var_cv[varnm]["long_name"])
    else:
        pca = xr.open_dataset("/data/met/processing/80_pca-anova/change-pca/"+varnm+"_change-pca_"+mtype+".nc").sel(season = season, period = period)
        cbar_label = var_cv[varnm]["plot_label"]
    
    if not cmap: cmap = var_cv[varnm]["cmap"]
        
    pca_plot(pca, cmap = cmap, contour = contour, contour_interval = contour_interval, markers = None, colours = None, fs = fs, cbar_label = cbar_label)
    fig = plt.gcf()

    if title: 
        fig.suptitle("Principal component analysis of change in "+season+" "+varnm+ " ("+period+")", fontweight = "bold")
        
    ve = [" ("+str(int(x.round()))+"%)" for x in pca.var_expl]
    
    fig.axes[0].set_title("Ensemble mean change", size = fs)
    fig.axes[1].set_title(mtype.upper()+" EPP1"+ve[1], size = fs)
    fig.axes[2].set_title(mtype.upper()+" EPP2"+ve[2], size = fs)
    
    if save:
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_pca-"+mtype+"-relative.png")
        else:
            plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_pca-"+mtype+".png")
        
        
#=================================================================================================================

def change_anova(varnm, season, period = "20491201-20791130", save = False, rel_diff = False, title = False, vmin = None, vmax = None, contour = False, contour_interval = 1, contour_offset = False):
    
    if rel_diff:
        da = xr.open_dataset("/data/met/processing/80_pca-anova/change-pca/"+varnm+"_change-anova-relative.nc").sel(season = season, period = period)[varnm]
        ss_cbar_label = "SD of relative change (%)"
    else:
        da = xr.open_dataset("/data/met/processing/80_pca-anova/change-pca/"+varnm+"_change-anova.nc").sel(season = season, period = period)[varnm]
        ss_cbar_label = var_cv[varnm]["plot_label"]
    
    vr = pd.read_csv("/data/met/processing/80_pca-anova/allvars-change-anova.csv")
    vr = vr[(vr.season == season) & (vr.varnm == varnm) & (vr.period == period)]
    def ve(v): return str(int((vr[v] * 100).round(0)))     # quick helper function to extract & format the variance explained
    
    if var_cv[varnm]["cmap"] == "RdBu":
        cmap = "Blues"
    elif var_cv[varnm]["cmap"] == "RdBu_r":
        cmap = "Reds"
    else:
        cmap = "plasma"
    
    map_args = {"contour" : contour, "contour_interval" : contour_interval, "contour_offset" : contour_offset}
    
    fig, axs = ukmap_subplots(ncols = 4, figsize = (15,5))
    cbar_ss = qmap(da.sel(src = "sd"), ax = axs[0], colorbar = False, vmin = vmin, vmax = vmax, cmap = cmap, title = "(a) Ensemble standard deviation\n", **map_args)
    
    cbar_ve = qmap(da.sel(src = "gcm"), ax = axs[1], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, 
                   title = "(b) Proportion explained by GCM\n("+ve("var_Gb")+"-"+ve("var_Ga")+"%)", **map_args)
    qmap(da.sel(src = "rcm"), ax = axs[2], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "(c) Proportion explained by RCM \n("+ve("var_Ra")+"-"+ve("var_Rb")+"%)", **map_args)
    qmap(da.sel(src = "res"), ax = axs[3], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "(d) Residual uncertainty \n("+ve("var_res")+"%)", **map_args)
    
    plt.colorbar(cbar_ss, ax = axs[0], location = "bottom", pad = 0.02, fraction = 0.05, label = ss_cbar_label)
    plt.colorbar(cbar_ve, ax = axs[1:], location = "bottom", pad = 0.02, fraction = 0.03, label = "% of variance explained by each component")
    
    if title:
        fig.suptitle("Contribution of each component to total variance in changes in "+season+" "+varnm+" in the UKCORDEX ensemble ("+period+")", fontweight = "bold")
    
    if save:
        if rel_diff:
            fig.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_change-anova-relative.png")
        else:
            fig.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_change-anova.png")


#=================================================================================================================

def change_sdmaps(varnm, season, period = "20491201-20791130", cmap = None, vmin = 0, vmax = None, rel_diff = False, recover_clim = False,
                  contour = False, contour_interval = 1, save = False, title = False, extend = "neither", ensembles = "all"):
    
    da = load_diff(varnm, season, period, rel_diff = rel_diff, recover_clim = recover_clim)
    sds = xr.concat([v.std("run").reset_coords(drop = True) for v in da.values()], "src").assign_coords(src = list(da.keys()))
    
    if not ensembles == "all":
        sds = sds.sel(src = ensembles)
    
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
    
    for i in range(len(sds.src)):
        cbar = qmap(sds.isel(src = i), ax = axs[i], colorbar = False, title = "("+string.ascii_lowercase[i]+") "+sds.src.values[i], 
                    vmin = vmin, vmax = vmax, cmap = cmap, contour = contour, contour_interval = contour_interval) 
    
    if rel_diff:
        cbar_label = "Standard deviation of relative change in "+re.sub("\(.+\)", "", lc(var_cv[varnm]["plot_label"])+" (%)")
    else: 
        cbar_label = "Standard deviation of "+lc(var_cv[varnm]["plot_label"])
    plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.035, pad = 0.01, label = cbar_label, extend = extend)
    
    if title:
        fig.suptitle("Standard deviation of biases in "+season+" "+varnm+" in each ensemble", fontweight = "bold")
    
    if save: plt.savefig("/data/met/reports/evaluation/plots/"+varnm+"_"+season+"_19890101-20081231_sd-maps.png", bbox_inches='tight')
        
        
#=================================================================================================================

def change_spreadplots(season, period = "20491201-20791130", recover_clim = True, sortby = None, minmax = False, inaline = False, width_only = False, save = False, blwidth = 1):
    
    if minmax:
        tas01 = load_diff("tasmin", season = season, period = period, recover_clim = recover_clim); tas01 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas01.items()}
        tas = load_diff("tas", season = season, period = period, recover_clim = recover_clim); tas = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas.items()}
        tas99 = load_diff("tasmax", season = season, period = period, recover_clim = recover_clim); tas99 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas99.items()}
        vnm = "tasmxx"
    else:
        tas01 = load_diff("tas01", season = season, period = period, recover_clim = recover_clim); tas01 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas01.items()}
        tas = load_diff("tas", season = season, period = period, recover_clim = recover_clim); tas = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas.items()}
        tas99 = load_diff("tas99", season = season, period = period, recover_clim = recover_clim); tas99 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas99.items()}
        vnm = "tasxx"
        
    if width_only:
        tas01 = {k : v - tas[k] for k, v in tas01.items()}
        tas99 = {k : v - tas[k] for k, v in tas99.items()}
        tas = {k : v - tas[k] for k, v in tas.items()}
    
    ymin = int(np.floor(min([t.min() for t in tas01.values()])))+1
    ymax = int(np.ceil(max([t.max() for t in tas99.values()])))+1
        
    if sortby == "bias":
        tas01 = {k : v if k == "obs" else v.sortby(tas[k], ascending = False) for k, v in tas01.items()}
        tas = {k : v if k == "obs" else v.sortby(tas[k], ascending = False) for k, v in tas.items()}
        tas99 = {k : v if k == "obs" else v.sortby(tas[k], ascending = False) for k, v in tas99.items()}
    if sortby == "rcm":
        tas01 = {k : v.sortby(v.run.str.replace(".+_","")) if "CORDEX" in k else v for k, v in tas01.items()}
        tas = {k : v.sortby(v.run.str.replace(".+_","")) if "CORDEX" in k else v for k, v in tas.items()}
        tas99 = {k : v.sortby(v.run.str.replace(".+_","")) if "CORDEX" in k else v for k, v in tas99.items()}
          
    if inaline:
        fig, axs = plt.subplots(ncols = 5, figsize = (25,5), sharey = True, dpi = 100, facecolor='w', edgecolor='k', gridspec_kw = {"width_ratios" : [1,1,3,1,1]})
        axlist = axs
    else:
        fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (16,10), sharey = True, dpi = 100, facecolor='w', edgecolor='k')
        gs = axs[0,3].get_gridspec()
        for ax in axs[0, :4]: ax.remove()
        axbig = fig.add_subplot(gs[0, :4])
        axlist = [axs[1,0], axs[1,1], axbig, axs[1,2], axs[1,3]]
    
    for j in range(5):
        ens_name = list(tas01.keys())[j]
        axz = axlist[j]
               
        t01 = list(tas01.values())[j]; t = list(tas.values())[j]; t99 = list(tas99.values())[j]
            
        if width_only:
            axz.axhline(0, color = "black", alpha = 0.3)
            [axz.axhline(x, color = "grey", alpha = 0.3) for x in np.sort(np.unique(np.concatenate([np.arange(0, ymax+blwidth, blwidth), np.arange(0, ymin*np.sign(ymin)+blwidth, blwidth)*np.sign(ymin)])))]
        else:
            [axz.axhline(x, color = "grey", alpha = 0.3) for x in np.arange(ymin, ymax+blwidth, blwidth)]
        axz.set_title(ens_name)
        
        for i in range(len(t01.run.values)):
            axz.plot(np.repeat(i,2), [t01.values[i], t99.values[i]], color = "black")
            rnm = t01.run.values[i]
            if "UKCP18" in ens_name:
                axz.plot(i, t.values[i], marker = "P", ms = 7, ls = "", color = "black")
                axz.set_xticks(range(len(t01.run.values)))
                axz.set_xticklabels(t01.run.values)
            elif ens_name == "ERA-EuroCORDEX":
                axz.plot(i, t.values[i], marker = "*", ms = 10, ls = "", color = rcm_colours[re.sub(".+_","",rnm)], mec = "black")
                axz.set_xticklabels("")
            elif ens_name == "EuroCORDEX":
                axz.plot(i, t.values[i], marker = gcm_markers[re.sub("p1_.+","p1",rnm)], ms = 9, ls = "", color = rcm_colours[re.sub(".+_","",rnm)], mec = "black")
                axz.set_xticklabels("")
            elif ens_name == "CMIP5-13":
                if rnm in cmip5_ec:
                    col = "white"
                else:
                    col = "grey"
                axz.plot(i, t.values[i], marker = gcm_markers[rnm], ms = 9, ls = "", color = col, mec = "black")
                axz.set_xticklabels("")
            elif ens_name == "CMIP5-EC":
                axz.plot(i, t.values[i], marker = gcm_markers[rnm], ms = 9, ls = "", color = "white", mec = "black")
                axz.set_xticklabels("")
        
        # add markers showing range without bias
        #if showranges:
        #    tc01, tc99 = [list(t.values())[j] - list(tas.values())[j] + tas["obs"].mean() for t in [tas01, tas99]]
        #    for i in range(len(tc01.run.values)):
        #        axz.plot(np.repeat(i,2), [tc01.values[i], tc99.values[i]], color = "darkred", alpha = 0.5, marker = "_", ls = "")
        
        if save:
            if width_only:
                plt.savefig("/data/met/reports/evaluation/change-plots/"+vnm+"_"+season+"_"+period+"_interval-plots.png", bbox_inches='tight')
            else:
                plt.savefig("/data/met/reports/evaluation/change-plots/"+vnm+"_"+season+"_"+period+"_spread-plots.png", bbox_inches='tight')
            
            
#=================================================================================================================

def change_varexpl(varnms, season, periods = "change", vtype = "var", ax = None, legend = False, rel_diff = False):
    
    if type(varnms) != list: varnms = [varnms]
    if rel_diff:
        df = pd.read_csv("/data/met/processing/80_pca-anova/allvars-change-anova-relative.csv", index_col = None)[::-1]
    else:
        df = pd.read_csv("/data/met/processing/80_pca-anova/allvars-change-anova.csv", index_col = None)[::-1]
        
    df = df[df.season == season]
    df = df[[v in varnms for v in df.varnm]]
    if periods == "all": 
        df = df
    elif periods == "change":
        ch_periods = ['20191201-20491130', '20291201-20591130', '20391201-20691130', '20491201-20791130']
        df = df[[p in ch_periods for p in df.period]]
    else:
        df = df[[p in periods for p in df.period]]
    
    df["period"] = [re.sub("1981-2011","1980-2010",str(int(sl[:4])+1)+"-"+str(int(sl[9:13])+1)) for sl in df.period.values]
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (6,len(df)/3), dpi= 100, facecolor='w', edgecolor='k')
    
    for i in range(len(df)):
        ve = df.iloc[i]
        ax.plot([ve[vtype+"_Ga"], ve[vtype+"_Gb"]], np.repeat(i,2), lw = 4, color = "red", alpha = 0.5)
        ax.plot([ve[vtype+"_Rb"], ve[vtype+"_Ra"]], np.repeat(i,2), lw = 4, color = "blue", alpha = 0.5)
        ax.plot(ve[vtype+"_res"], i, marker = "o", ms = 4, color = "black", ls = "", alpha = 0.5)
        ax.plot(ve[vtype+"_G2"], i, marker = "x", ms = 4, color = "darkred", ls = "")
        ax.plot(ve[vtype+"_R2"], i, marker = "x", ms = 4, color = "darkblue", ls = "")
    
    if vtype == "dev":
        ax.set_xlabel("% deviance explained by each component")
    else:
        ax.set_xlabel("% uncertainty explained by each component")
        
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,len(df["varnm"] + " " + df["period"])-.5)
    ax.set_yticks(list(range(len(df))))
    ax.set_xticklabels(range(0,101,20))
    for v in np.arange(0,1,0.1): ax.axvline(v, color = "grey", alpha = 0.1, zorder = -10)
    if len(varnms) == 1:
        ax.set_yticklabels(df["period"])
    else:
        ax.set_yticklabels(df["varnm"] + " " + df["period"])
    
    if legend: 
        plt.legend(["Range of GCM contribution", "Range of RCM contribution", "Residual uncertainty", "Unbalanced GCM contribution", "Unbalanced RCM contribution"], edgecolor = "white",
                   loc = 'upper left', bbox_to_anchor = (1, 0.6))
        
#=================================================================================================================

def change_stampplot(varnm, season, period = "20491201-20791130", recover_clim = False, rel_diff = False, ensemble = "EuroCORDEX", bias = True, cmap = None, save = False, title = False):
    
    # load the data
    data = load_diff(varnm, season, period = period, recover_clim = recover_clim, rel_diff = rel_diff)
    ens_list = list(data.values())
    ens_names = list(data.keys())
       
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
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = var_cv[varnm]["plot_label"])
    
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
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = var_cv[varnm]["plot_label"])
    
    else:
        
        # Otherwise, plot all the EuroCORDEX runs, plus their driving GCMs
        fig, axs = ukmap_subplots(ncols = 10, nrows = 11, figsize = (23,24))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title:
            fig.suptitle("Change in "+season+" "+varnm+" for EuroCORDEX RCMs and CMIP5 driving RCMs", fontweight = "bold", y = 1)
            fig.subplots_adjust(top = 0.96)
        
        vlims = vrange(xr.concat([ens_list[ens_names.index("CMIP5-EC")], ens_list[ens_names.index("EuroCORDEX")]], "new"))
            
        # could filter out any unwanted GCMs here as well
        cmip_gcms = [x for x in list(gcm_full_names.keys())]
        ec_gcms = [x for x in list(gcm_full_names.values())]
            
        for gcm_nm in cmip5_ec:
            if gcm_nm in ens_list[ens_names.index("CMIP5-EC")].run.values:
                cbar = qmap(ens_list[ens_names.index("CMIP5-EC")].sel(run = gcm_nm), ax = axs[0,cmip_gcms.index(gcm_nm)], cmap = cmap, **vlims, colorbar = False)
        
        for rcm_nm in ens_list[ens_names.index("EuroCORDEX")].run.values:
            rcm_i = list(rcm_colours.keys()).index(re.sub(".+_", "", rcm_nm))
            gcm_i = ec_gcms.index(re.sub("p1_.+","p1",rcm_nm))
            qmap(ens_list[ens_names.index("EuroCORDEX")].sel(run = rcm_nm), ax = axs[rcm_i+1,gcm_i], cmap = cmap, **vlims, colorbar = False)
    
        # add GCM & RCM names to first row & column. Currently impossible to add a ylabel to cartopy axes.
        for i in range(10): axs[0,i].set_title(re.sub("_", "\n",cmip_gcms[i]))
        for j in range(10): axs[j+1,0].text(-0.07, 0.55, list(rcm_colours.keys())[j], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[j+1,0].transAxes, fontsize = "large")
        axs[0,0].text(-0.07, 0.55, "GCM", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0,0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.01, pad = 0.01, label = var_cv[varnm]["plot_label"])
    
    if save: plt.savefig("/data/met/reports/evaluation/change-plots/"+varnm+"_"+season+"_"+period+"_"+ensemble.lower()+"_stampplot.png")
        

#=================================================================================================================



def trend_boxplots(dat, ax = None, j = 0.025, biaslines = "auto", blwidth = 1, hline = 0, ukcp_en = False, legend_pos = "best", **bp_kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (8,5), dpi = 100, facecolor = "w", edgecolor = "black")
        
    dat = dat.assign_coords(period = [str(int(p[:4])+1)+"-"+str(int(p[9:13])+1) if not p == "19801201-20101130" else str(int(p[:4]))+"-"+str(int(p[9:13])) for p in dat.period.values])
               
    offsets = dict(zip(list(gcm_markers.keys())[1:11], list(np.arange(j*-4.5, j*5.5, j).round(3))))
    
    ax.violinplot(dat, showextrema = False)
    ax.boxplot(dat, labels = dat.period.str.replace("-", "-\n").values, medianprops = {"color":"black"}, showmeans = True, meanline = True, meanprops = {"color":"black"}, flierprops = {"marker":""}, **bp_kwargs)
    
    if biaslines == "auto":
        bl_vals = np.array([v for v in np.concatenate([m.values for m in dat]) if not np.isnan(v)])
        biaslines = [int(np.floor(bl_vals.min())), int(np.ceil(bl_vals.max()))]
    
    if biaslines is not None:
        for i in np.arange(biaslines[0], biaslines[1] + blwidth, blwidth):
            ax.axhline(i, zorder = -99, color = "grey", alpha = 0.5, linewidth = 0.5)
    
    if hline is not None:
        ax.axhline(hline, color = "darkred", ls = "--")
    
    if dat.run.str.contains(".+p1_.+").all():
        # EuroCORDEX
        for p in range(len(dat.period)):
            mscatter([offsets[n]+p+1 for n in dat.run.str.replace("p1_.+", "p1").values], dat.isel(period = p), m = run_markers(dat), c = run_colours(dat), edgecolor = "black", zorder  = 99, ax = ax)
    
    elif "01" in dat.run.values:
        # UKCP18
        if ukcp_en:
            for p in range(len(dat.period)):
                mscatter(np.repeat(p+1, len(dat.run)), dat.isel(period = p), m = ["$"+r+"$" for r in dat.run.values], c = list(rcm_colours.values()), zorder  = 99, ax = ax, s = 60)
        else:
            mscatter(np.tile(range(len(dat.period)+1)[1:], (1, len(dat.run))), dat, ax = ax, m = ".", color = "black", edgecolor = "black", zorder = 98)
            mscatter(range(len(dat.period)+1)[1:], dat.sel(run = "01"), ax = ax, m = "o", color = "orange", edgecolor = "black", zorder = 99)
    
    elif dat.run.str.contains(".+p1").all():
        # CMIP5
        for p in range(len(dat.period)):
            mscatter(np.repeat(p+1, len(dat.run)), dat.isel(period = p), m = run_markers(dat), c = gcm_colours(dat.run.values), edgecolor = "black", zorder  = 99, ax = ax)

    else:
        # Need to code points for other ensembles
        mscatter(np.tile(range(len(dat.period)+1)[1:], (1, len(dat.run))), dat, ax = ax, m = ".", color = "black", edgecolor = "black", zorder = 98)
        
    # legend for boxplots
    bplot_handles = [matplotlib.lines.Line2D([], [], color = "black", linestyle = "-"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "--")]
    bplot_labels = ["Group median", "Group mean"]
    ax.legend(handles = bplot_handles, labels = bplot_labels, edgecolor = "white", loc = legend_pos)
        

#=================================================================================================================