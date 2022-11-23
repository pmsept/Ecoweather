import xarray as xr
import numpy as np
import pandas as pd

import sys
sys.path.append('/data/met/processing/10_methods')
from regridding import vec2map, map2vec
from plotting import *

import matplotlib.pyplot as plt


def svd(da, dim = "run", n_pcs = 2, scores = True):
    
    # Method to compute PC decomposition of array over a selected dimension
    # da is assumed to contain anomalies
    
    if n_pcs is None: n_pcs = len(da[dim])
    pc_labels = ["pc" + str(i) for i in list(range(n_pcs+1))[1:]]
    
    # flatten array & drop any cells with NA values (trims to common grid)
    # get mean map to reconstruct from vectors later
    da_map = da.mean(dim)
    
    # flatten array & drop any cells with NA values (trims to common grid)
    da_stack = map2vec(da)
    
    extra_dims = [d for d in da_stack.dims if not d in [dim, "s"]]

    if len(extra_dims) > 0:
        
        extra_dim_values = pd.DataFrame(pd.core.reshape.util.cartesian_product([da[d].values for d in extra_dims])).rename(index = dict(zip(range(len(extra_dims)), extra_dims))).to_dict()
        
        all_ds = []
        
        for i in range(len(extra_dim_values)):
            ys = da_stack.sel(**extra_dim_values[i])
            tss = sum(np.diag(ys.values @ ys.values.transpose()))
            
            u, s, vh = np.linalg.svd(ys.values, full_matrices = False)
            vh = xr.concat([vec2map(vh[n,:], da_map.sel(**extra_dim_values[i])) for n in range(n_pcs)], dim = "pc").assign_coords(pc = (["pc"], pc_labels)).expand_dims(extra_dims)
            ds = xr.Dataset(data_vars = { "svecs" : (vh.dims, vh.data),
                                          "svals" : (["pc"], s[:n_pcs]),
                                          "scores" : (["run", "pc"], u[:,:n_pcs]),
                                          "var_expl" : (["pc"], s[:n_pcs]**2 / tss * 100)},
                            coords = vh.coords)
            all_ds.append(ds)
        
        ds = xr.combine_by_coords(all_ds)
        
    else: 
        
        tss = sum(np.diag(da_stack.values @ da_stack.values.transpose()))
        
        # singular value decomposition
        u, s, vh = np.linalg.svd(da_stack.values, full_matrices = False)
        
        vh = xr.concat([vec2map(vh[i,:], da_map) for i in range(n_pcs)], dim = "pc").assign_coords(pc = (["pc"], pc_labels))
        ds = xr.Dataset(data_vars = { "svecs" : (vh.dims, vh.data),
                                      "svals" : (["pc"], s[:n_pcs]),
                                      "scores" : (["run", "pc"], u[:,:n_pcs]),
                                      "var_expl" : (["pc"], (svals**2) / tss * 100)},
                        coords = svecs.coords)
            
    # sign correction: ensure that all svecs are positive on average
    sc_adj = np.sign(ds.svecs.mean(["projection_x_coordinate", "projection_y_coordinate"]))
    ds["svecs"] = ds["svecs"] * sc_adj
    ds["scores"] = ds["scores"] * sc_adj
    ds = ds.assign_coords(run = da.run.values)
    
    if not scores:
        ds = ds.drop_vars(["scores", "run"])
        
    return ds



def pc_modes(svd):
    
    return svd.svecs * svd.svals



def pc_scores(svd, fields):
    
    svd_stack = map2vec(svd.svecs / svd.svals)
    f_stack = map2vec(fields)
    
    # check if extra dimensions to iterate over
    extra_dims = [d for d in svd_stack.dims if not d in ["pc", "s"]]
    if len(extra_dims) > 0:
        extra_dim_values = pd.DataFrame(pd.core.reshape.util.cartesian_product([svd_stack[d].values for d in extra_dims])).rename(index = dict(zip(range(len(extra_dims)), extra_dims))).to_dict()
        
        all_sc = []
        for i in range(len(extra_dim_values)):
            all_sc.append(svd_stack.sel(**extra_dim_values[i]) @ f_stack.sel(**extra_dim_values[i]).expand_dims(extra_dims))
            
        scores = xr.combine_by_coords(all_sc)
        
    else:
        scores = svd_stack @ f_stack

    return scores
    
    
    
def ss(da):
    
    ystack = da.stack(s = (["projection_x_coordinate", "projection_y_coordinate"])).dropna("s", "any")
    
    if "season" in da.dims:
        tss = [sum(np.diag(ystack.sel(season = seas).values @ ystack.sel(season = seas).values.transpose())) for seas in da.season.values]
    else: 
        tss = sum(np.diag(ystack.values @ ystack.values.transpose()))
        
    return tss


def pc_anova(varnm, season, period = "19890101-20081231", split_pcs = True):
    
    # get variance decomposition (for balanced designs only)
    pca = xr.open_dataset("/data/met/processing/80_results/evaluation-pca/pca_"+varnm+"_"+period+"_balanced.nc").sel(season = season)
    
    if split_pcs:
        var_decomp = (xr.concat([(pca.var_expl.sel(pc = [pc for pc in pca.pc.values if mtype in pc]) * pca[mtype+"_ss"]  / pca.tss) for mtype in ["gcm", "rcm", "run"]], "pc")).values.tolist()
    else:
        var_decomp = [(s / pca.tss * 100).values.tolist() for s in [pca.gcm_ss, pca.rcm_ss, pca.run_ss]]
    
    return var_decomp


def pc_varexp(varnm, season, period = "19890101-20081231"):
    
    pca = xr.open_dataset("/data/met/processing/80_results/evaluation-pca/pca_"+varnm+"_"+period+"_balanced.nc").sel(season = season)
    return dict(zip(pca.pc.values.tolist()[1:], pca.var_expl.values.tolist()[1:]))    



def squares(da):
    
    # Compute sums of squares for a given group & return as map
    
    # flatten array & drop any cells with NA values (trims to common grid)
    da_stack = map2vec(da)
    extra_dims = [d for d in da_stack.dims if not d in ["run", "s"]]
    
    if len(extra_dims) > 0:
        
        extra_dim_values = pd.DataFrame(pd.core.reshape.util.cartesian_product([da[d].values for d in extra_dims])).rename(index = dict(zip(range(len(extra_dims)), extra_dims))).to_dict()
        all_sq = []
        for i in range(len(extra_dim_values)):
            ys = da_stack.sel(**extra_dim_values[i])
            sq = vec2map(np.diag(ys.values.transpose() @ ys.values), da.sel(**extra_dim_values[i]).mean("run")).expand_dims(extra_dims).rename("squares")
            all_sq.append(sq)
            
        if len(extra_dims) == 1:
            all_sq = xr.concat(all_sq, extra_dims[0])
        else:
            all_sq = xr.combine_by_coords(all_sq).squares
    else:
        all_sq = vec2map(np.diag(da_stack.values.transpose() @ da_stack.values), da.mean("run")).rename("squares")
        
    return all_sq



def svd_trunc(mat, compute_uv = True):
    
    # SVD, returning only first r components where r is matrix rank
    mrank = np.linalg.matrix_rank(mat)
    if compute_uv:
        u, s, v = np.linalg.svd(mat, full_matrices = False, compute_uv = True)
        return u[:mrank,:mrank], s[:mrank], v[:mrank,:]
    else:
        s = np.linalg.svd(mat, full_matrices = False, compute_uv = False)
        return s[:mrank]

    



def svd_inv(da):
    
    # compute generalised inverse of sum of squares using SV decomposition
    u, s, v = svd_trunc(da.values)   
    return v.transpose() @ np.diag(s**(-2)) @ v


def svd_squares(da):
    
    # compute matrix of cross products using SV decomposition (no rank truncation)
    u, s, v = np.linalg.svd(da, full_matrices = False)
    
    return v.transpose() @ np.diag(s**2) @ v




def manova_ts(lam):
    
    # compute test statistics from vector of eigenvalues
    return { "Wilks" : np.prod([1 / (1+l) for l in lam]), "Pillai" : np.sum([l/(1+l) for l in lam]), "Hotelling-Lawley": np.sum(lam), "Roy" : lam.max() }


def manova_es(lam, N = 42):
    
    # compute effect sizes from vector of eigenvalues + total sample size N 
    df = len(lam)
    L, U, V, R = manova_ts(lam).values()
    
    return { "Partial eta^2" : 1 - L, "Partial omega^2" : 1 - (N*L) / (N - df - 1 + L), "tau^2" : 1 - L**(1/df), "xi^2" : U/df, "zeta^2" : V / (df + V) }



def deviance_explained(Sigma_inv, squares):
    
    # method to compute scaled deviances per REC method
    traces = [sum(np.diag((Sigma_inv @ s))) for s in squares]
    return [tr / traces[0] for tr in traces]


def variance_explained(squares):
    
    ve = [sum(np.diag(ss)) / sum(np.diag(squares[0])) for ss in squares]
    return ve



def deviance(da, ybar, gcm_bar, rcm_bar, res, n_rcms = None, n_gcms = None, Sigma_inv = None):
    
    if n_rcms is None: n_rcms = np.sqrt(len(rcm_bar))
    if n_gcms is None: n_gcms = np.sqrt(len(gcm_bar))
        
    if n_gcms is None and n_rcms is None: 
        df = (G-1)*(R-1)
    else:
        df = 1
    
    T = svd_squares(map2vec(da - ybar))
    T_G = svd_squares(map2vec((gcm_bar - ybar) * n_rcms))
    T_R = svd_squares(map2vec((rcm_bar - ybar) * n_gcms))
    T_E = svd_squares(map2vec(res))
    
    # optionally, can fix Sigma_inv
    if Sigma_inv is None: 
        Sigma_inv = svd_inv(map2vec(res)) * df
    
    traces = [sum(np.diag((Sigma_inv @ s))) for s in [T, T_G, T_R, T_E]]
    
    return traces


def gcm_means(da):
    gcm_names = da.run.str.replace("i1p1_.+", "i1p1")
    return da.groupby(gcm_names).mean(skipna = False).reindex(run = gcm_names).assign_coords(run = da.run)

def rcm_means(da):
    rcm_names = da.run.str.replace(".+_", "")
    return da.groupby(rcm_names).mean(skipna = False).reindex(run = rcm_names).assign_coords(run = da.run)




def qpca(da, decomp_by = "run", n_pcs = 5):
    
    # quick PCA decomposition without groups
    ybar = da.mean(decomp_by, skipna = False)
    
    u, s, vh = np.linalg.svd(map2vec(da - ybar).values, full_matrices = False)
    tss = sum(np.diag(vh.transpose() @ np.diag(s**2) @ vh))
    
    svecs = xr.concat([ybar] + [vec2map(vh[n,:], ybar) for n in range(n_pcs)], dim = "pc")
    
    ds = xr.Dataset(data_vars = { "svecs"    : (svecs.dims, svecs.data),
                                 "svals"    : (["pc"], np.append(np.array(1), s[:n_pcs])),
                                 "var_expl" : (["pc"], np.append(np.array(1), ((s**2) / tss * 100)[:n_pcs])),
                                 "scores"   : ([decomp_by, "pc"], np.column_stack([np.ones(u[:,1].shape), u[:,:n_pcs]]))},
                    coords = svecs.coords).assign_coords({decomp_by : da[decomp_by].values, "pc" : ["mean"] + list(range(n_pcs+1)[1:])})
    
    # sign correction: ensure that all svecs are positive on average, with the exception of the mean
    sc_adj = np.sign(ds.svecs.mean(["projection_x_coordinate", "projection_y_coordinate"])); sc_adj[0] = 1
    ds["svecs"] = ds["svecs"] * sc_adj
    ds["scores"] = ds["scores"] * sc_adj
    
    return ds



def qpca_plot(pca, cmap = "RdBu", markers = None, colours = None, cbar_label = None, **kwargs):
    
    # quick plots of mean + first two PCs of output from `qpca`
    modes = pca.svecs * pca.svals
    
    fig, axs = ukmap_subplots(ncols = 5, figsize = (15,4), gridspec_kw = {'width_ratios' : [1,1,1,0.01,1.5]})
    vlims = vrange(modes[:3])
    
    ve = [" "+str(int(x.round()))+"%" for x in pca.var_expl.sel(pc = ["1","2"])]
    
    cbar = qmap(modes.sel(pc = "mean"), ax = fig.axes[0], title = "Overall bias", **vlims, colorbar = False, cmap = cmap, **kwargs)
    qmap(modes.sel(pc = "1"), ax = fig.axes[1], title = "PC1: "+ve[0], **vlims, colorbar = False, cmap = cmap, **kwargs)
    qmap(modes.sel(pc = "2"), ax = fig.axes[2], title = "PC2: "+ve[1], **vlims, colorbar = False, cmap = cmap, **kwargs)
    
    plt.colorbar(cbar, ax = fig.axes[0:3], location = "bottom", pad = 0.02, fraction = 0.05, label = cbar_label)
    
    if markers is None and "run" in pca.dims:
        if "01" in pca.run.values:
            markers = ["$"+n+"$" for n in pca.run.values]
        elif "i1p1" in pca.run.values[0]:
            markers = [gcm_markers[rnm] for rnm in pca.run.values]
        else:
            markers = "o"
            
    if colours is None and "run" in pca.dims:
        if "01" in pca.run.values:
            colours = [rcm_colours["HadREM3-GA7-05"] if r == "01" else "white" for r in pca.run.values]
        elif "i1p1" in pca.run.values[0]:
            colours = "black"
        else:
            colours = [rcm_colours[rnm] for rnm in pca.run.values]
        
    ax3 = fig.add_subplot(144)        
    mscatter(pca.scores.sel(pc = "1"), pca.scores.sel(pc = "2"), ax = ax3, m = markers, c = colours, s = 60, edgecolor = "black", zorder = 9)
    
    # axes & labels
    ax3.set_xlim(*vrange(pca.scores.sel(pc = "1").values * 1.2).values())
    ax3.set_ylim(*vrange(pca.scores.sel(pc = "2").values * 1.2).values())
    ax3.axvline(0, linestyle = "--", color = "grey")
    ax3.axhline(0, linestyle = "--", color = "grey")
    ax3.set_xlabel("First PC score")
    ax3.set_ylabel("Second PC score")
