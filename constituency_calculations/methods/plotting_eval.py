
import xarray as xr
import pandas as pd
import glob
import re

import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/met/processing/10_methods')
from taylorDiagram import TaylorDiagram
from misc import load_eval, xy
from dicts import *
from plotting import *
from regridding import *

# default to tight bounding box
matplotlib.rcParams['savefig.bbox'] = "tight"

from IPython.display import clear_output


import warnings
warnings.filterwarnings("ignore", message = "FixedFormatter should only be used together with FixedLocator")  # something about adding vertical gridlines to variance explained


#####################################################################################################

def clim_biasmaps(varnm, season, period = "19890101-20081231", rel_diff = False, obs_cmap = None, contour = False, contour_interval = 1, contour_offset = False, save = False, title = False, cities = None, obs_vmin = None, obs_vmax = None, remove_13 = False, vmin = None, vmax = None):
    
    # load the data
    data = load_eval(varnm, season, period)
    if remove_13:
        data = {re.sub("CMIP5-EC", "CMIP5", k):v for k, v in data.items() if not k == "CMIP5-13"}
    obs = list(data.values())[0]
    ens_list = list(data.values())[1:]
    ens_names = list(data.keys())[1:]
    
    if rel_diff:
        ens_biases = [((ens - obs) / obs * 100).mean("run", skipna = False) for ens in ens_list]
        cbar_label = "% difference in " + lc(var_cv[varnm]["plot_label"])
    else:
        ens_biases = [(ens - obs).mean("run", skipna = False) for ens in ens_list]
        cbar_label = "Difference in " + lc(var_cv[varnm]["plot_label"])
    
    # set colour maps
    if not obs_cmap: obs_cmap = var_cv[varnm]["cmap"]
    bias_cmap = var_cv[varnm]["cmap"]
    
    vlims = vrange(xr.concat(ens_biases, "src"))
    if vmin is not None : vlims["vmin"] = vmin
    if vmax is not None : vlims["vmax"] = vmax
    
    fig, axs = ukmap_subplots(ncols = len(data), figsize = (len(data)*3 - 2,5))
    if title:
        fig.suptitle("Observed "+season+" "+varnm+" and mean bias per ensemble", fontweight = "bold")
        
    map_kwargs = { "contour" : contour, "contour_interval" : contour_interval, "contour_offset" : contour_offset, "cities" : cities }
    
    cbar_obs = qmap(obs, ax = axs[0], colorbar = False, title = "(a) HadUK-grid (observed) " + varnm, label = var_cv[varnm]["plot_label"], cmap = obs_cmap, **map_kwargs, vmin = obs_vmin, vmax = obs_vmax)
    
    for i in range(len(ens_biases)):
        cbar_bias = qmap(ens_biases[i], ax = axs[i+1], colorbar = False, title = "("+string.ascii_lowercase[i+1]+") "+ens_names[i]+ " bias", **vlims, cmap = bias_cmap, **map_kwargs)
    
    print(cbar_pars)
    plt.colorbar(cbar_obs, ax = axs[0], location = "bottom", fraction = 0.05, pad = 0.01, label = var_cv[varnm]["plot_label"])
    plt.colorbar(cbar_bias, ax = axs[1:], location = "bottom", fraction = 0.05, pad = 0.01, label = cbar_label)
    
    if save: 
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_bias-maps-relative.png", bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_bias-maps.png", bbox_inches='tight')    
            
            
#=================================================================================================================

def clim_boxplots(varnm, season, period = "19890101-20081231", bias = False, rel_diff = False, model_legend = False, legend_w = 0.65, save = False, title = False, j = 0.025, jitter_ukcp = False,
                  ax = None, biaslines = "auto", blwidth = 1, remove_13 = False):
    
    # load the data
    data = load_eval(varnm, season, period)
    if remove_13: data = {re.sub("CMIP5-EC", "CMIP5", k):v for k, v in data.items() if not k == "CMIP5-13"}
    obs = list(data.values())[0]
    ens_list = list(data.values())[1:]
    ens_names = list(data.keys())[1:]
    
    ylabel = var_cv[varnm]["plot_label"]
    hline = 0
    xy = ["projection_x_coordinate", "projection_y_coordinate"]
    
    if varnm == "prcprop":
        ens_list = [da for da, n in zip(ens_list, ens_names) if not n == "UKCP18 60km"]
        ens_names = [n for n in ens_names if not n == "UKCP18 60km"]
        
    if rel_diff:
        means = [((ens.mean(xy) - obs.mean(xy)) / obs.mean(xy) * 100) for ens in ens_list]
        ylabel = "% bias in " + lc(var_cv[varnm]["plot_label"])
    else:
        if bias:
            means = [(ens - obs).mean(xy) for ens in ens_list]
            b = "bias in "
        else:
            means = [ens.mean(xy) for ens in ens_list]
            hline = obs.mean()
            b = ""
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if ax is None:
        if model_legend:
            fig, ax = plt.subplots(1,1, figsize = (12,6), dpi= 100, facecolor='w', edgecolor='k')
        else:
            fig, ax = plt.subplots(1,1, figsize = (8,6), dpi= 100, facecolor='w', edgecolor='k')
    if title: fig.suptitle("Boxplots of mean "+b+season+" "+varnm+" over the UK for each model", fontweight = "bold")
    
    ax.violinplot(means, showextrema = False)
    ax.boxplot(means, labels = [re.sub(" ", "\n", re.sub("ERA-","ERA-\n",n)) for n in ens_names], meanline = True, showmeans = True, zorder = 3, 
               meanprops = {"color" : "black"}, medianprops = {"color" : "black"}, flierprops = {"marker" : ""})
    
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
        if jitter_ukcp:
            ukcp_x = [i+x+1 for x in np.arange(0.025*-5.5, 0.025*6, 0.025)]
            m = ["$"+rn+"$" for rn in means[i].run.values]
        else:
            ukcp_x = np.repeat(i+1, len(means[i]))
            m = "o"
        mscatter(ukcp_x, means[i], color = "black", m = m, edgecolor = "k", zorder = 5, s = 60, ax = ax)
        mscatter(ukcp_x[0], means[i][0], color = rcm_colours["HadREM3-GA7-05"], marker = m[0], edgecolor = "k", zorder = 5, s = 60, ax = ax)
    
    if "UKCP18 12km" in ens_names:
        i = ens_names.index("UKCP18 12km")
        if jitter_ukcp:
            ukcp_x = [i+x+1 for x in np.arange(0.025*-5.5, 0.025*6, 0.025)]
            m = ["$"+rn+"$" for rn in means[i].run.values]
        else:
            ukcp_x = np.repeat(i+1, len(means[i]))
            m = "o"
        mscatter(ukcp_x, means[i], color = "black", m = m, edgecolor = "k", zorder = 5, s = 60, ax = ax)
        mscatter(ukcp_x[0], means[i][0], color = rcm_colours["HadREM3-GA7-05"], marker = m[0], edgecolor = "k", zorder = 5, s = 60, ax = ax)
    
    if biaslines == "auto":
        bl_vals = np.concatenate([m.values for m in means])
        biaslines = [int(np.floor(bl_vals.min() - hline)), int(np.ceil(bl_vals.max() - hline))]
    
    if biaslines is not None:
        if varnm == "prcprop":
            for i in range(0,100,10):
                ax.axhline(i, zorder = -99, color = "grey", alpha = 0.5, linewidth = 0.5)
            ax.set_ylim(0,100)
            hline = np.nan
        else:
            for i in np.arange(biaslines[0], biaslines[1], blwidth):
                ax.axhline(hline + i, zorder = -99, color = "grey", alpha = 0.5, linewidth = 0.5)
            ax.set_ylim(hline + biaslines[0], hline + biaslines[1])
    
    ax.axhline(hline, color = "darkred", linestyle = "--")

    # y-axis label. No titles because these will be added in the main document.
    ax.set_ylabel(ylabel)
    
    if model_legend:    
        # legend for RCM colours & GCM marker styles
        plt.subplots_adjust(right = legend_w)
        rcm_legend(set(rcm_rcms), ax = plt.gcf(), loc = 'upper left', bbox_to_anchor = (legend_w, 0.9))
        gcm_legend(list(gcm_markers.keys())[1:11], ax = plt.gcf(), loc = 'lower left', bbox_to_anchor = (legend_w, 0.1))
    
    # legend for boxplots
    bplot_handles = [matplotlib.lines.Line2D([], [], color = "darkred", linestyle = "--"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "-"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "--")]
    bplot_labels = ["HadUK-grid", "Group median", "Group mean"]
    ax.legend(handles = bplot_handles, labels = bplot_labels, edgecolor = "white")
    
    if save: 
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_boxplots-relative.png", bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_boxplots.png", bbox_inches='tight')

            
#=================================================================================================================

def Taylor_plot(varnm, season, period = "19890101-20081231", max_sd = None, save = False, legend = False, title = False, rect = 111, fig = None, extend = False):
    
    # load the data
    data = load_eval(varnm, season, period)
    obs = list(data.values())[0]
    ens_list = list(data.values())[1:]
    ens_names = list(data.keys())[1:]
    
    # compute correlations & SDs to get range of axes
    corrs = [xr.corr(ens, obs, ["projection_y_coordinate", "projection_x_coordinate"]).values for ens in ens_list]
    sds = [ens.std(["projection_y_coordinate", "projection_x_coordinate"]).values for ens in ens_list]
    ref_sd = obs.std(["projection_y_coordinate", "projection_x_coordinate"]).values
    if max_sd == None:
        max_sd = np.ceil(max(np.concatenate(sds)) / ref_sd)
    else:
        max_sd = max_sd / ref_sd
    
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
    
    td = TaylorDiagram(ref_sd, fig = fig, label = "HadUK-grid sd", srange = (0,max_sd), extend = extend, rect = rect)
    td.add_contours(colors = "mistyrose")
    td.add_grid(color = "papayawhip")
    if extend: plt.axvline(0, color = "black", lw = 1, ls = "--")
    
    if title:
        plt.suptitle("Taylor diagram evaluating spatial patterns of "+season+" "+varnm+" with respect to HadUK-Grid observations", fontweight = "bold")
    
    if legend:
        rcm_legend(rcm_names = list(rcm_colours.keys())[:12], loc='upper left', bbox_to_anchor = (0.95, 0.95))
        leg = gcm_legend(gcm_names = list(gcm_markers.keys())[:12], loc='lower left', bbox_to_anchor = (0.95, 0.1))
    
    if "CMIP5-13" in ens_names:
        for r in ens_list[ens_names.index("CMIP5-13")].run.values:
            ens = ens_list[ens_names.index("CMIP5-13")].sel(run = r)
            td.add_sample(ens.std().values, xr.corr(ens, obs), marker = gcm_markers[r], label = "_", ms = 7, ls = '', mfc = gcm_colours([r]), mec = "black", zorder = 10)
    
    if "CMIP5-EC" in ens_names:
        for r in ens_list[ens_names.index("CMIP5-EC")].run.values:
            ens = ens_list[ens_names.index("CMIP5-EC")].sel(run = r)
            td.add_sample(ens.std().values, xr.corr(ens, obs), marker = gcm_markers[r], label = "_", ms = 7, ls = '', mfc = gcm_colours([r]), mec = "black", zorder = 11)
    
    if "EuroCORDEX" in ens_names:
        for r in ens_list[ens_names.index("EuroCORDEX")].run.values:
            ens = ens_list[ens_names.index("EuroCORDEX")].sel(run = r)
            td.add_sample(ens.std().values, xr.corr(ens, obs), marker = gcm_markers[re.sub("p1_.+", "p1", r)], label = "_", ms = 7, ls = '', mfc = rcm_colours[re.sub(".+_", "", r)], mec="black", zorder = 13)
    
    if "ERA-EuroCORDEX" in ens_names:
        for r in ens_list[ens_names.index("ERA-EuroCORDEX")].run.values:
            ens = ens_list[ens_names.index("ERA-EuroCORDEX")].sel(run = r)
            td.add_sample(ens.std().values, xr.corr(ens, obs), marker = "*", label = "_", ms = 7, ls = '', mfc = rcm_colours[re.sub(".+_", "", r)], mec="black", zorder = 12)
            
    if "UKCP18 60km" in ens_names:
        for r in ens_list[ens_names.index("UKCP18 60km")].run.values:
            if r == 1:
                mcolour = rcm_colours["HadREM3-GA7-05"]
            else:
                mcolour = "black"
            ens = ens_list[ens_names.index("UKCP18 60km")].sel(run = r)
            td.add_sample(ens.std().values, xr.corr(ens, obs), marker = "x", label = "_", ms = 7, ls = '', mfc = mcolour, mec = mcolour, zorder = 14)
    
    if "UKCP18 12km" in ens_names:
        for r in ens_list[ens_names.index("UKCP18 12km")].run.values:
            if r == 1:
                mcolour = rcm_colours["HadREM3-GA7-05"]
            else:
                mcolour = "black"
            ens = ens_list[ens_names.index("UKCP18 12km")].sel(run = r)
            td.add_sample(ens.std().values, xr.corr(ens, obs), marker = "+", label = "_", ms = 7, ls = '', mfc = mcolour, mec = mcolour, zorder = 15)
        
    if save: 
        if legend:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_Taylor-diagram.png", bbox_extra_artists=(leg), bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_Taylor-diagram.png", bbox_inches='tight')
            

#=================================================================================================================

def bias_pca(varnm, season, mtype = "gcm", cmap = None, contour = False, contour_interval = 1, save = False, title = False, rel_diff = False, fs = 12):
    
    if rel_diff:
        pca = xr.open_dataset("/data/met/processing/80_pca-anova/bias-pca/"+varnm+"_bias-pca_"+mtype+"-relative.nc").sel(season = season)
        cbar_label = "Difference in "+lc(var_cv[varnm]["long_name"])
    else:
        pca = xr.open_dataset("/data/met/processing/80_pca-anova/bias-pca/"+varnm+"_bias-pca_"+mtype+".nc").sel(season = season)
        cbar_label = var_cv[varnm]["plot_label"]
    
    if not cmap: cmap = var_cv[varnm]["cmap"]
        
    pca_plot(pca, cmap = cmap, contour = contour, contour_interval = contour_interval, markers = None, colours = None, fs = fs, cbar_label = cbar_label)
    fig = plt.gcf()

    if title: 
        fig.suptitle("Principal component analysis of bias in "+season+" "+varnm+ " (19890101-20081231)", fontweight = "bold")
        
    ve = [" ("+str(int(x.round()))+"%)" for x in pca.var_expl]
    
    fig.axes[0].set_title("Ensemble mean bias", size = fs)
    fig.axes[1].set_title(mtype.upper()+" EPP1"+ve[1], size = fs)
    fig.axes[2].set_title(mtype.upper()+" EPP2"+ve[2], size = fs)
    
    if save: 
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_19890101-20081231_pca-"+mtype+"-relative.png")
        else:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_19890101-20081231_pca-"+mtype+".png")
        

#=================================================================================================================

def bias_anova(varnm, season, save = False, title = False, sd_vmin = None, sd_vmax = None, rel_diff = False, contour = False, contour_interval = 1, contour_offset = False):
    
    if rel_diff:
        da = xr.open_dataset("/data/met/processing/80_pca-anova/bias-pca/"+varnm+"_bias-anova-relative.nc").sel(season = season)[varnm]
        vr = pd.read_csv("/data/met/processing/80_pca-anova/allvars-bias-anova-relative.csv")
    else:
        da = xr.open_dataset("/data/met/processing/80_pca-anova/bias-pca/"+varnm+"_bias-anova.nc").sel(season = season)[varnm]
        vr = pd.read_csv("/data/met/processing/80_pca-anova/allvars-bias-anova.csv")

    vr = vr[(vr.season == season) & (vr.varnm == varnm)]
    
    def ve(v): return str(int((vr[v] * 100).round(0)))     # quick helper function to extract & format the variance explained
    
    if var_cv[varnm]["cmap"] == "RdBu":
        cmap = "Blues"
    elif var_cv[varnm]["cmap"] == "RdBu_r":
        cmap = "Reds"
    else:
        cmap = "plasma"
        
    qmap_kwargs = {"contour" : contour, "contour_interval" : contour_interval, "contour_offset" : contour_offset}
    
    fig, axs = ukmap_subplots(ncols = 4, figsize = (15,5))
    cbar_ss = qmap(da.sel(src = "sd"), ax = axs[0], colorbar = False, cmap = cmap, title = "(a) Ensemble standard deviation\n", vmin = sd_vmin, vmax = sd_vmax, **qmap_kwargs)
    
    cbar_ve = qmap(da.sel(src = "gcm"), ax = axs[1], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "(b) Proportion explained by GCM\n("+ve("var_Gb")+"-"+ve("var_Ga")+"%)", **qmap_kwargs)
    qmap(da.sel(src = "rcm"), ax = axs[2], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "(c) Proportion explained by RCM \n("+ve("var_Ra")+"-"+ve("var_Rb")+"%)", **qmap_kwargs)
    qmap(da.sel(src = "res"), ax = axs[3], colorbar = False, cmap = "viridis", vmin = 0, vmax = 100, title = "(d) Residual uncertainty \n("+ve("var_res")+"%)", **qmap_kwargs)
    
    plt.colorbar(cbar_ss, ax = axs[0], location = "bottom", pad = 0.02, fraction = 0.05, label = var_cv[varnm]["plot_label"])
    plt.colorbar(cbar_ve, ax = axs[1:], location = "bottom", pad = 0.02, fraction = 0.03, label = "% of variance explained by each component")
    
    if title:
        fig.suptitle("Contribution of each component to total variance in biases in "+season+" "+varnm+" in the UKCORDEX ensemble ("+design+" design)", fontweight = "bold")
    
    if save:
        if rel_diff:
            fig.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_19890101-20081231_bias-anova-relative.png")
        else:
            fig.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_19890101-20081231_bias-anova.png")
        

#=================================================================================================================

def bias_varexpl(varnms, season, vtype = "var", ax = None, legend = False, rel_diff = False):
    
    if type(varnms) != list: varnms = [varnms]
    
    if rel_diff:
        df = pd.read_csv("/data/met/processing/80_pca-anova/allvars-bias-anova-relative.csv", index_col = None)
    else:
        df = pd.read_csv("/data/met/processing/80_pca-anova/allvars-bias-anova.csv", index_col = None)
    df = df[df.season == season]
    df = df[[v in varnms for v in df.varnm]]

    if ax is None:
        fig, ax = plt.subplots(figsize = (6,len(df)/3), dpi= 100, facecolor='w', edgecolor='k')
    
    for i in range(len(df)):
        ve = df[df.varnm == varnms[i]]
        ax.plot([ve[vtype+"_Ga"].values, ve[vtype+"_Gb"].values], np.repeat(i,2), lw = 4, color = "red", alpha = 0.5)
        ax.plot([ve[vtype+"_Rb"].values, ve[vtype+"_Ra"].values], np.repeat(i,2), lw = 4, color = "blue", alpha = 0.5)
        ax.plot(ve[vtype+"_res"].values, i, marker = "o", ms = 4, color = "black", ls = "", alpha = 0.5)
        ax.plot(ve[vtype+"_G2"].values, i, marker = "x", ms = 4, color = "darkred", ls = "")
        ax.plot(ve[vtype+"_R2"].values, i, marker = "x", ms = 4, color = "darkblue", ls = "")
    
    if vtype == "dev":
        ax.set_xlabel("% deviance explained by each component")
    else:
        ax.set_xlabel("% uncertainty explained by each component")
        
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,len(df)-.5)
    ax.set_yticks(list(range(len(df))))
    ax.set_xticklabels(range(0,101,20))
    for v in np.arange(0,1,0.1): ax.axvline(v, color = "grey", alpha = 0.1, zorder = -10)
    ax.set_yticklabels(varnms)
    
    if legend: 
        plt.legend(["Range of GCM contribution", "Range of RCM contribution", "Residual uncertainty", "Unbalanced GCM contribution", "Unbalanced RCM contribution"], edgecolor = "white",
                   loc = 'center left', bbox_to_anchor = (1.05, 0.35))

#=================================================================================================================

def bias_stampplot(varnm, season, period = "19890101-20081231", ensemble = "EuroCORDEX", bias = True, cmap = None, save = False, title = False, rel_diff = False):
    
    # load the data
    data = load_eval(varnm, season, period)
    obs = list(data.values())[0]
    ens_list = list(data.values())[1:]
    ens_names = list(data.keys())[1:]
    
    if rel_diff:
        ens_list = [(ens - obs) / obs * 100 for ens in ens_list]
        b = "bias-"
        bt = "Relative bias in climatology"
        ct = "Relative bias in "+lc(var_cv[varnm]["plot_label"])
    elif bias:
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
    
    if save: 
        if rel_diff:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_"+ensemble.lower()+"_"+b+"stampplot-relative.png")
        else:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+varnm+"_"+season+"_"+period+"_"+ensemble.lower()+"_"+b+"stampplot.png")
        

#=================================================================================================================

def bias_spreadplots(season, centre = False, sortby = None, minmax = False, inaline = False, showranges = True, save = False):
    
    if minmax:
        tas01 = load_eval("tasmin", season); tas01 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas01.items()}
        tas = load_eval("tas", season); tas = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas.items()}
        tas99 = load_eval("tasmax", season); tas99 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas99.items()}
        vnm = "tasmxx"
    else:
        tas01 = load_eval("tas01", season); tas01 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas01.items()}
        tas = load_eval("tas", season); tas = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas.items()}
        tas99 = load_eval("tas99", season); tas99 = {k : v.mean(["projection_x_coordinate", "projection_y_coordinate"]) for k, v in tas99.items()}
        vnm = "tasxx"
    
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
        fig, axs = plt.subplots(ncols = 6, figsize = (25,5), sharey = True, dpi = 100, facecolor='w', edgecolor='k', gridspec_kw = {"width_ratios" : [1,1,3,1,1,1]})
        axlist = axs
    else:
        fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (16,10), sharey = True, dpi = 100, facecolor='w', edgecolor='k')
        gs = axs[0,3].get_gridspec()
        for ax in axs[0, :3]: ax.remove()
        axbig = fig.add_subplot(gs[0, :3])
        axlist = [axs[1,0], axs[1,1], axbig, axs[0,3], axs[1,2], axs[1,3]]
    
    for j in range(6):
        ens_name = list(tas01.keys())[j+1]
        axz = axlist[j]
               
        if centre:
            t = list(tas.values())[j+1]
            t01 = list(tas01.values())[j+1] - t; t99 = list(tas99.values())[j+1] - t; t = t - t
            [axz.axhline(da["obs"].mean() - tas["obs"].mean(), ls = "--", color = "darkred", alpha = 0.7) for da in [tas01, tas, tas99]]
            [axz.axhline(x - tas["obs"].mean(), color = "grey", alpha = 0.3) for x in range(ymin, ymax)]
            showranges = False
        else:
            t01 = list(tas01.values())[j+1]; t = list(tas.values())[j+1]; t99 = list(tas99.values())[j+1]
            [axz.axhline(da["obs"].mean(), ls = "--", color = "darkred", alpha = 0.7) for da in [tas01, tas, tas99]]
            [axz.axhline(x, color = "grey", alpha = 0.3) for x in range(ymin, ymax)]

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
        if showranges:
            tc01, tc99 = [list(t.values())[j+1] - list(tas.values())[j+1] + tas["obs"].mean() for t in [tas01, tas99]]
            for i in range(len(tc01.run.values)):
                axz.plot(np.repeat(i,2), [tc01.values[i], tc99.values[i]], color = "darkred", alpha = 0.5, marker = "_", ls = "")
        
        if save:
            plt.savefig("/data/met/reports/evaluation/eval-plots/"+vnm+"_"+season+"_19890101-20081231_spread-plots.png", bbox_inches='tight')
                
        #axz.set_xticks(range(len(t01.run.values)))
        #axz.set_xticklabels(t01.run.values, rotation = 90)
        

#=================================================================================================================

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
        

#####################################################################################################

def prcprop_boxplots(season, period = "19890101-20081231", save = False, j = 0.025, ax = None, biaslines = "auto", blwidth = 10, recover_clim = True):
    
    # slightly adapted method to account for absence of obs & UKCP18 60kmm runs in prcprop
    # load the data
    if period == "19890101-20081231":
        data = load_eval("prcprop", season, "19890101-20081231")
        obs = list(data.values())[0]
        ens_list = list(data.values())[1:]
        ens_names = list(data.keys())[1:]
    else:
        data = load_diff("prcprop", season, period = period, recover_clim = recover_clim)
        ens_list = list(data.values())
        ens_names = list(data.keys())

    
    # filter out UKCP18 60km, for which convective precipitation is unavailable
    ens_list = [da for da, n in zip(ens_list, ens_names) if not n == "UKCP18 60km"]
    ens_names = [n for n in ens_names if not n == "UKCP18 60km"]
        
    xy = ["projection_x_coordinate", "projection_y_coordinate"]
    
    means = [(ens).mean(xy) for ens in ens_list]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (8,6), dpi= 100, facecolor='w', edgecolor='k')
    
    ax.violinplot(means, showextrema = False)
    ax.boxplot(means, labels = [re.sub("ERA-EuroCORDEX","ERA-RCM",n) for n in ens_names], meanline = True, showmeans = True, zorder = 3, meanprops = {"color" : "black"}, medianprops = {"color" : "black"}, flierprops = {"marker" : ""})
    
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
    
    if "UKCP18 12km" in ens_names:
        i = ens_names.index("UKCP18 12km")
        ax.scatter(np.repeat(i+1, len(means[i])), means[i], color = "black", marker = ".", edgecolor = "k", zorder = 5, s = 60)
        ax.scatter(i+1, means[i][0],
                   color = rcm_colours["HadREM3-GA7-05"], marker = ".", edgecolor = "k", zorder = 5, s = 60)
    
    if biaslines == "auto":
        bl_vals = np.concatenate([m.values for m in means])
        biaslines = [int(np.floor(bl_vals.min())), int(np.ceil(bl_vals.max()))+blwidth]
    
    if biaslines is not None:
        for i in np.arange(biaslines[0], biaslines[1], blwidth):
            ax.axhline(i, zorder = -99, color = "grey", alpha = 0.5, linewidth = 0.5)
        ax.set_ylim(biaslines[0], biaslines[1])
    
    # y-axis label. No titles because these will be added in the main document.
    ax.set_ylabel("Percentage of total precipitation arising from convective processes")
    
    # legend for boxplots
    bplot_handles = [matplotlib.lines.Line2D([], [], color = "black", linestyle = "-"),
                     matplotlib.lines.Line2D([], [], color = "black", linestyle = "--")]
    bplot_labels = ["Group median", "Group mean"]
    ax.legend(handles = bplot_handles, labels = bplot_labels, edgecolor = "white")
    
    if save: 
        if period == "19890101-20081231":
            plt.savefig("/data/met/reports/evaluation/eval-plots/prcprop_"+season+"_19890101-20081231_boxplots.png", bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/change-plots/prcprop_"+season+"_"+period+"_boxplots.png", bbox_inches='tight')
        

#=================================================================================================================

def prcprop_biasmaps(season, period = "19890101-20081231", contour = False, contour_interval = 1, contour_offset = False, save = False, recover_clim = True):
    
    varnm = "prcprop"
    

    # load the data
    if period == "19890101-20081231":
        data = load_eval("prcprop", season, "19890101-20081231")
        obs = list(data.values())[0]
        ens_list = list(data.values())[1:]
        ens_names = list(data.keys())[1:]
    else:
        data = load_diff("prcprop", season, period = period, recover_clim = recover_clim)
        ens_list = list(data.values())
        ens_names = list(data.keys())
    
    # filter out UKCP18 60km, for which convective precipitation is unavailable
    ens_list = [da for da, n in zip(ens_list, ens_names) if not n == "UKCP18 60km"]
    ens_names = [n for n in ens_names if not n == "UKCP18 60km"]
    
    ens_biases = [ens.mean("run", skipna = False) for ens in ens_list]
    
    # set colour maps
    bias_cmap = "RdBu"
        
    fig, axs = ukmap_subplots(ncols = len(ens_names), figsize = (len(ens_names)*3 - 2,5))
    
    map_kwargs = { "contour" : contour, "contour_interval" : contour_interval, "contour_offset" : contour_offset}
        
    for i in range(len(ens_biases)):
        cbar_bias = qmap(ens_biases[i], ax = axs[i], colorbar = False, title = "("+string.ascii_lowercase[i]+") "+ens_names[i], vmin = 0, vmax = 100, cmap = bias_cmap, **map_kwargs)
    
    plt.colorbar(cbar_bias, ax = axs, location = "bottom", fraction = 0.03, pad = 0.01, label = "Percentage of precipitation due to convective processes")
    
    if save: 
        if period == "19890101-20081231":
            plt.savefig("/data/met/reports/evaluation/eval-plots/prcprop_"+season+"_"+period+"_bias-maps.png", bbox_inches='tight')
        else:
            plt.savefig("/data/met/reports/evaluation/change-plots/prcprop_"+season+"_"+period+"_bias-maps.png", bbox_inches='tight')
        

#=================================================================================================================

def prcprop_stampplots(season, period = "20491201-20791130", ensemble = "EuroCORDEX", cmap = None, save = False, title = False):
    
    varnm = "prcprop"
    
    # load the data
    data = load_diff(varnm = varnm, season = season, period = period, recover_clim = True)
    ens_list = list(data.values())
    ens_names = list(data.keys())
    
    main = "Average "+season+" proportion of convective precipitation over the UK from "+period[:4]+"-"+period[9:13]
    cbar_label = var_cv[varnm]["plot_label"]
    
    if not cmap: cmap = var_cv[varnm]["cmap"]
    vlims = { "vmin" : 0, "vmax" : 100 }
    
    if ensemble == "CMIP5":
        
        # Assuming no GCM data exists for EC-EARTH r3i1p1: will need to rejig if it becomes available
        fig, axs = ukmap_subplots(ncols = 9, nrows = 2, figsize = (16,6))
        fig.subplots_adjust(hspace = 0.25, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title: 
            fig.subplots_adjust(top = 0.85)
            fig.suptitle(main+" for CMIP5 GCMs", fontweight = "bold", y = 1)
                
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
        
        fig, axs = ukmap_subplots(ncols = 12, nrows = 1, figsize = (20,6))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title: 
            fig.subplots_adjust(top = 0.93)
            fig.suptitle(main+" for UKCP18 global & regional runs", fontweight = "bold", y = 1)
                
        for i in range(12):
            cbar = qmap(ens_list[ens_names.index("UKCP18 12km")].isel(run = i), ax = axs[i], cmap = cmap, **vlims, colorbar = False, title = ens_list[ens_names.index("UKCP18 12km")].isel(run = i).run.values.tolist())
            
        axs[0].text(-0.07, 0.55, "12km", va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform = axs[0].transAxes, fontsize = "large")
        
        plt.colorbar(cbar, ax = axs, location = "bottom", fraction = 0.04, pad = 0.01, label = cbar_label)
    
    else:
        
        # Otherwise, plot all the EuroCORDEX runs, plus their driving GCMs
        fig, axs = ukmap_subplots(ncols = 10, nrows = 11, figsize = (20,24))
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 1, left = 0.01, right = 0.99, bottom = 0.02)
        
        if title:
            fig.suptitle(main+" for EuroCORDEX RCMs and CMIP5 driving RCMs", fontweight = "bold", y = 1)
            fig.subplots_adjust(top = 0.96)
                
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
    
    if save: plt.savefig("/data/met/reports/evaluation/plots/"+varnm+"_"+season+"_"+period+"_"+ensemble.lower()+"_stampplot.png")
        