import xarray as xr
import re
import glob

import matplotlib.pyplot as plt

import sys
sys.path.append('/data/met/processing/10_methods')
from regridding import *
from pc_decomposition import *

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
ro.numpy2ri.activate() 
r = ro.r

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Methods for ordered decomposition of variance & deviance, based on code provided by Richard Chandler
# in emails between November 23rd & December 7th 2021

# Support functions

def MakeXXinvX(X):
    return (X @ np.diag(1/X.sum(0)).transpose()).transpose()

def MakeHat(X):
    return X @ MakeXXinvX(X).transpose()

def TEinvT(SVD_E, SVD_M):
    
    # get ranks & remove redundant elements
    rE = sum(np.abs(SVD_E[1]) > 1e-12)
    rM = sum(np.abs(SVD_M[1]) > 1e-12)
    
    sE = SVD_E[1][:rE]; VE = SVD_E[2][:rE,:].transpose()
    sM = SVD_M[1][:rM]; VM = SVD_M[2][:rM,:].transpose()
    
    LamVVE = np.diag(sM) @ VM.transpose() @ VE
    
    DE = sum(np.diag(np.diag(1/sE**2) @ LamVVE.transpose() @ LamVVE))
    return DE


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Model 2 & EPPs
    
def fit_model2(da):
            
    # split full run name into GCM and RCM
    r.assign('gcm', da.run.str.replace("p1_.+","p1").values)
    r.assign('rcm', da.run.str.replace(".+_","").values)
    
    # convert to R factor
    r('gcm <- as.factor(gcm)')
    r('rcm <- as.factor(rcm)')
    
    # create design matrix in R & pass back to numpy array for use in python methods
    X2 = np.array(r('model.matrix(~ gcm + rcm, contrasts.arg = list(gcm = "contr.sum", rcm = "contr.sum"))'))
    
    # use ybar to reconstruct maps
    ybar = da.mean("run", skipna = False)
    
    M = map2vec(da).values
    theta_hat = np.linalg.solve(X2.transpose() @ X2, X2.transpose() @ M)
    
    G = r('gcm').max()
    
    mu_hat = theta_hat[0,:]
    alpha_hat = np.row_stack([theta_hat[1:G,:], -theta_hat[1:G,:].sum(0)])
    beta_hat = np.row_stack([theta_hat[G:,:], -theta_hat[G:,:].sum(0)])
    res_2 = M - (X2 @ theta_hat)
    
    return { "mu_hat" : mu_hat, "alpha_hat" : alpha_hat, "beta_hat" : beta_hat, "res_2" : res_2, "gcms" : list(r('levels(gcm)')), "rcms" : list(r('levels(rcm)'))}


def get_epps(effects, reshape_to = None, n_pcs = None):
    
    # SVD over fitted effects
    u, s, vh = np.linalg.svd(effects, full_matrices = False)
    
    # compute total sum of squares
    tss = sum(np.diag(vh.transpose() @ np.diag(s**2) @ vh))
    
    # compile singular vectors into DataArray for easier storage 
    if reshape_to is not None:
        svecs = xr.concat([vec2map(vh[n,:], reshape_to) for n in range(len(s))], dim = "pc")
    else:
        svecs = xr.DataArray(data = vh, dims = ["pc", "S"])
    
    # combine
    ds = xr.Dataset(data_vars = {"svecs"    : (svecs.dims, svecs.data),
                                 "svals"    : (["pc"], s),
                                 "var_expl" : (["pc"], (s**2) / tss * 100),
                                 "scores"   : (["run", "pc"], u)},
                    coords = svecs.coords).assign_coords(pc = list(range(len(s))))
    
    # if not all pcs are required, truncate
    if n_pcs is not None:
        ds = ds.sel(pc = range(n_pcs+1)[1:])
    
    # correct signs of PCs so that positive score always denotes an overall increase
    sc_adj = np.sign(svecs.mean([d for d in svecs.dims if d != "pc"]))
    sc_adj[0] = 1
    ds["svecs"] = ds["svecs"] * sc_adj
    ds["scores"] = ds["scores"] * sc_adj
    
    return ds


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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ANOVA for models 1a, 1b and 2

def unbalanced_anova(da):
        
    # split run names into GCM and RCM
    r.assign('gcm', da.run.str.replace("p1_.+","p1").values)
    r.assign('rcm', da.run.str.replace(".+_","").values)
    
    # convert lists of GCMs and RCMS to factors
    r('gcm <- as.factor(gcm)')
    r('rcm <- as.factor(rcm)')
    
    # construct design matrices using R, pass back to python
    XG = np.array(r('model.matrix(formula(~ gcm - 1))'))
    XR = np.array(r('model.matrix(formula(~ rcm - 1))'))
    X2 = np.array(r('model.matrix(~ gcm + rcm, contrasts.arg = list(gcm = "contr.sum", rcm = "contr.sum"))'))
    
    # centre the data
    ybar = da.mean("run", skipna = False)
    M = map2vec(da - ybar).values
    
    # get GCM/RCM effects and residuals from model 2
    theta_hat = np.linalg.solve(X2.transpose() @ X2, X2.transpose() @ M)
    
    G = XG.shape[1]
    MG_2 = XG @ np.row_stack([theta_hat[1:G,:], -theta_hat[1:G,:].sum(0)])
    MR_2 = XR @ np.row_stack([theta_hat[G:,:], -theta_hat[G:,:].sum(0)])
    res_2 = M - (X2 @ theta_hat)
    
    # (X'X)^{-1}X' and hat matrices for 1-way models
    XXXG = (XG @ np.diag(1/XG.sum(0)).transpose()).transpose()
    XXXR =  (XR @ np.diag(1/XR.sum(0)).transpose()).transpose()
    
    HG = XG @ XXXG
    HR = XR @ XXXR
    
    # scaled GCM and RCM means
    MG_a = (np.diag(np.sqrt(XG.sum(0))) @ XXXG) @ M
    MR_b = (np.diag(np.sqrt(XR.sum(0))) @ XXXR) @ M
    
    # get partial residuals & use to compute second part of explained variance
    # (could just do this by subtraction but that seems like more work & computation doesn't take long)
    pres_1a = M - (HG @ M)
    pres_1b = M - (HR @ M)
    
    MG_b = pres_1b - res_2
    MR_a = pres_1a - res_2
    
    # use SVD to compute traces & trace props, deviances & deviance proportions
    svds = { src : np.linalg.svd(X) for src, X in { "total":M, "Ga":MG_a, "Gb":MG_b, "G2" : MG_2, "Ra":MR_a, "Rb":MR_b, "R2" : MR_2, "res":res_2 }.items() }

    traces = { k : sum(v[1]**2) for k, v in svds.items() }
    deviances = {k : TEinvT(svds["res"], svd) for k, svd in svds.items()}
    
    props = { "var_"+k : v / traces["total"] for k, v in traces.items() if not k == "total" }
    props.update({ "dev_"+k : dev / deviances["total"] for k, dev in deviances.items() if not k == "total" })
    
    # standard deviations are computed over trimmed data (ie. with values > 5 IQR above q99 removed())
    ve_maps = [vec2map(np.diag(M_X.transpose() @ M_X) / np.diag(M.transpose() @ M) * 100, ybar) for M_X in [MG_2, MR_2, res_2]]
    ve_maps = xr.concat([trim_da(da).std("run", skipna = True)] + ve_maps, "src").assign_coords(src = ["sd", "gcm", "rcm", "res"])

    return props, ve_maps



def estimate_anova(da):
    
    # Estimate anova for filled ensemble
    
    # get ensemble SD
    ens_sd = da.std("run", skipna = False)
    
    # get fitted effects
    m2_fit = fit_model2(da)
    
    # numbers of runs (total / GCM / RCM)
    n = len(da.run)
    G = len(m2_fit["gcms"])
    R = len(m2_fit["rcms"])   
    
    # estimate 'complete' SSCPs using reweighted fitted effects & residuals
    TG_est = R * m2_fit["alpha_hat"].transpose() @ m2_fit["alpha_hat"]
    TR_est = G * m2_fit["beta_hat"].transpose() @ m2_fit["beta_hat"]
    TE_est = (R-1) * (G-1) / (n - R - G - 1) * (m2_fit["res_2"].transpose() @ m2_fit["res_2"])
    T_est = TG_est + TR_est + TE_est
    
    # reshape into maps of proportion of T_est explained
    est_ve_maps = xr.concat([ens_sd] + [vec2map(np.diag(T / T_est * 100), ens_sd) for T in [TG_est, TR_est, TE_est]], "src").assign_coords(src = ["ens_sd", "gcm", "rcm", "res"])
    props = {src : sum(np.diag(T)) / sum(np.diag(T_est)) for src, T in {"var_G*" : TG_est, "var_R*" : TR_est, "var_res*" : TE_est}.items()}
    
    return props, est_ve_maps