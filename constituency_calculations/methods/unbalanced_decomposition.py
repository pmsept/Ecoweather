import xarray as xr
import re
import glob

import matplotlib.pyplot as plt

import sys
sys.path.append('/data/met/processing/10_methods')
from regridding import *
from pc_decomposition import *

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

####################################################################################################################
# Model 2 & EPPs
    
def fit_model2(da):
    
    # CONSTRUCT DESIGN MATRICES
    run_names = da.run.values
    gcms = sorted(set(da.run.str.replace("p1_.+","p1").values))
    rcms = sorted(set(da.run.str.replace(".+_","").values))
    
    n = len(run_names); G = len(gcms); R = len(rcms)
    
    # create matrices indicating group membership & stack to form X
    X_G = np.column_stack([[-1 if gcms[-1] in run_name else (1 if g in run_name else 0) for run_name in run_names] for g in gcms[:-1]])
    X_R = np.column_stack([[-1 if rcms[-1] in run_name else (1 if r in run_name else 0) for run_name in run_names] for r in rcms[:-1]])
    X = np.column_stack([np.ones([n, 1]), X_G, X_R])
    
    # ESTIMATE FITTED EFFECTS
    
    # flatten DataArray and remove all NA valueS
    Y = da.stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any").values

    # solve X'X𝜃 = X'Y to find fitted effects 𝜃
    theta_hat = np.linalg.solve(X.transpose() @ X, X.transpose() @ Y)
    
    # unpack the fitted coefficients and expand to obtain the Gth and Rth fitted effects
    mu_hat = theta_hat[0,:]
    alpha_hat = np.row_stack([theta_hat[1:G,:], -theta_hat[1:G,:].sum(0)])
    beta_hat = np.row_stack([theta_hat[G:,:], -theta_hat[G:,:].sum(0)])
    res = Y - (X @ theta_hat)
    
    
    return { "mu_hat" : mu_hat, "alpha_hat" : alpha_hat, "beta_hat" : beta_hat, "res" : res, "gcms" : gcms, "rcms" : rcms}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compute EPPs over fitted effects

def EPPs(effects, n_pcs = None, effect_labels = None, reshape_to = None):
    
    # SVD over fitted effects
    u, s, vh = np.linalg.svd(effects, full_matrices = False)
    
    # compute total sum of squares
    tss = sum(np.diag(vh.transpose() @ np.diag(s**2) @ vh))
    
    # compile singular vectors into DataArray for easier storage 
    if reshape_to is not None:
        svecs = xr.concat([vec2map(vh[n,:], reshape_to) for n in range(len(s))], dim = "pc")
    else:
        svecs = xr.DataArray(data = vh, dims = ["pc", "S"])
    
    # combine into DataSet
    ds = xr.Dataset(data_vars = {"svecs"    : (svecs.dims, svecs.data),
                                 "svals"    : (["pc"], s),
                                 "var_expl" : (["pc"], (s**2) / tss * 100),
                                 "scores"   : (["run", "pc"], u)},
                    coords = svecs.coords).assign_coords(pc = list(range(len(s))))
    
    # correct signs of PCs so that positive score always denotes an overall increase
    sc_adj = np.sign(svecs.mean([d for d in svecs.dims if d != "pc"]))
    sc_adj[0] = 1
    ds["svecs"] = ds["svecs"] * sc_adj
    ds["scores"] = ds["scores"] * sc_adj
    
    # if not all pcs are required, truncate
    if n_pcs is not None:
        ds = ds.sel(pc = range(n_pcs+1)[1:])
        
    if effect_labels is not None:
        ds = ds.assign_coords(run = effect_labels)
    
    return ds

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# analysis of variance for unbalanced ensemble

def unbalanced_manova(da):
    
    # GET ENSEMBLE MEAN & STANDARD DEVIATION
    Ybar = da.mean("run", skipna = False)
    ens_sd = da.std("run", skipna = False)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## MODEL 2 EFFECTS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONSTRUCT DESIGN MATRICES FOR MODEL 2 (GCM & RCM EFFECTS)
    run_names = da.run.values
    gcms = sorted(set(da.run.str.replace("p1_.+","p1").values))
    rcms = sorted(set(da.run.str.replace(".+_","").values))
    
    n = len(run_names); G = len(gcms); R = len(rcms)
    
    # create matrices indicating group membership & stack to form X
    X_G = np.column_stack([[-1 if gcms[-1] in run_name else (1 if g in run_name else 0) for run_name in run_names] for g in gcms[:-1]])
    X_R = np.column_stack([[-1 if rcms[-1] in run_name else (1 if r in run_name else 0) for run_name in run_names] for r in rcms[:-1]])
    X = np.column_stack([np.ones([n, 1]), X_G, X_R])
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESTIMATE MODEL 2 EFFECTS
    
    # flatten DataArray and remove all NA valueS
    Y = da.stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any").values

    # solve X'X𝜃 = X'Y to find fitted effects 𝜃
    theta_hat = np.linalg.solve(X.transpose() @ X, X.transpose() @ Y)
    
    ahat_2 = np.row_stack([theta_hat[1:G,:], -theta_hat[1:G,:].sum(0)])
    bhat_2 = np.row_stack([theta_hat[G:,:], -theta_hat[G:,:].sum(0)])
    res_2 = Y - (X @ theta_hat)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PARTITIONING OF VARIANCE IN 'COMPLETED' ENSEMBLE
    
    # estimate SSCPs for 'completed' emsemble using reweighted fitted effects & residuals
    Tc_G2 = R * ahat_2.transpose() @ ahat_2
    Tc_R2 = G * bhat_2.transpose() @ bhat_2
    Tc_res = (R-1) * (G-1) * res_2.transpose() @ res_2 / (n - R - G - 1)
    Tc = Tc_G2 + Tc_R2 + Tc_res
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## MODEL 1 EFFECTS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # EFFECTS FOR MODELS 1a & 1b
    
    # fitted effects Y_.g and Y_r.
    ahat_1a = da.groupby(da.run.str.replace("p1_.+", "p1")).mean() - Ybar
    bhat_1b = da.groupby(da.run.str.replace(".+p1_", "")).mean() - Ybar
    
    # get fitted values for each run under each model
    fitted_1a = Ybar + ahat_1a.reindex(run = da.run.str.replace("p1_.+", "p1")).assign_coords(run = da.run.values)
    fitted_1b = Ybar + bhat_1b.reindex(run = da.run.str.replace(".+p1_", "")).assign_coords(run = da.run.values)

    # residual matrices for models 1a & 1b
    res_1a = (da - fitted_1a).stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any").values
    res_1b = (da - fitted_1b).stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any").values
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # COMPUTE PARTITIONING OF OBSERVED VARIANCE FOR EACH MODEL
    
    # residual matrix for model 0
    res_0 = (da - Ybar).stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any").values
    T_obs = res_0.transpose() @ res_0
    T_res = res_2.transpose() @ res_2
    
    T_Ga = T_obs - (res_1a.transpose() @ res_1a)
    T_Ra = (res_1a.transpose() @ res_1a) - T_res
    
    T_Rb = T_obs - (res_1b.transpose() @ res_1b)
    T_Gb = (res_1b.transpose() @ res_1b) - T_res
    
    T_G2 = (X_G @ ahat_2[:-1,:]).transpose() @ (X_G @ ahat_2[:-1,:])
    T_R2 = (X_R @ bhat_2[:-1,:]).transpose() @ (X_R @ bhat_2[:-1,:])
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## COMPUTE TRACES & RETURN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # compute ranges of proportion of variance explained by each component in the observed ensemble
    obs_ve = {k : (np.trace(v / np.trace(T_obs) * 100)).round(3) for k,v in zip(["ve_Ga", "ve_Ra", "ve_Gb", "ve_Rb", "ve_G2", "ve_R2", "ve_res"], [T_Ga, T_Ra, T_Gb, T_Rb, T_G2, T_R2, T_res])}
    
    # compute ranges of proportion of variance explained by each component in the completed ensemble
    est_ve = {k : (np.trace(v) / np.trace(Tc) * 100).round(3) for k,v in zip(["estve_G2", "estve_R2", "estve_res"], [Tc_G2, Tc_R2, Tc_res])}
    
    # maps of variance explained by each component in the completed ensemble
    est_ve_maps = [vec2map(np.diag(v / Tc * 100), Ybar) for v in [Tc_G2, Tc_R2, Tc_res]]
    est_ve_maps = xr.concat([ens_sd] + est_ve_maps, "src").assign_coords(src = ["sd", "gcm", "rcm", "res"])
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    return {"obs_ve" : obs_ve, "est_ve" : est_ve, "est_ve_maps" : est_ve_maps}