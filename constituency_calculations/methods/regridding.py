# Methods to compute vertices on regular grid, & (optionally) transform to a different coordinate reference system

import numpy as np
import xesmf as xe
import xarray as xr
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import geopandas
import regionmask
import pandas as pd
import glob
from sklearn.cluster import DBSCAN

from geopy.geocoders import Nominatim

from IPython.display import clear_output

# vertices for pcolormesh on OSGB grid
xv = list(range(-216000, 774000, 12000))
yv = list(range(-108000, 1237000, 12000))

# Dictionary of native CRS for each model
crs = { "CNRM-ALADIN63"                : ccrs.LambertConformal(central_longitude = 10.5, central_latitude = 49.5, standard_parallels = [49.5,]),
        "CLMcom-ETH-COSMO-crCLIM-v1-1" : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "DMI-HIRHAM5"                  : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "MOHC-HadREM3-GA7-05"          : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "SMHI-RCA4"                    : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "KNMI-RACMO22E"                : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "GERICS-REMO2015"              : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "ICTP-RegCM4-6"                : ccrs.LambertConformal(central_longitude = 9.75, central_latitude = 48.0, standard_parallels = [30., 65.], false_easting = -6000, false_northing = -6000, 
                                                               globe = ccrs.Globe(semimajor_axis = 6371229.0, inverse_flattening = None)),
        "CLMcom-CCLM4-8-17"            : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "IPSL-WRF381P"                 : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "latlon"                       : ccrs.RotatedPole(central_rotated_longitude = 180),
        "rotated_latlon"               : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "osgb"                         : ccrs.OSGB(approx = False),
        "ukcp18"                       : ccrs.RotatedPole(pole_longitude = 198.0, pole_latitude = 39.25, globe = ccrs.Globe(semimajor_axis = 6371229.0)),
        "uk_map"                       : ccrs.TransverseMercator(approx = False, central_longitude = -2, central_latitude = 49, scale_factor = 0.9996012717,
                                                                 false_easting = 400000, false_northing = -100000, globe = ccrs.Globe(datum = 'OSGB36', ellipse = 'airy')),
        "ALADIN63"                : ccrs.LambertConformal(central_longitude = 10.5, central_latitude = 49.5, standard_parallels = [49.5,]),
        "COSMO-crCLIM-v1-1" : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "HIRHAM5"                  : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "HadREM3-GA7-05"          : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "RCA4"                    : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "RACMO22E"                : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "REMO2015"              : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "RegCM4-6"                : ccrs.LambertConformal(central_longitude = 9.75, central_latitude = 48.0, standard_parallels = [30., 65.], false_easting = -6000, false_northing = -6000, 
                                                               globe = ccrs.Globe(semimajor_axis = 6371229.0, inverse_flattening = None)),
        "CCLM4-8-17"            : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
        "WRF381P"                 : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
      }

def get_crs(fnm):
    
    if "60km" in fnm or "cmip5" in fnm:
        return crs["latlon"]
    elif "ukcp18" in fnm:
        return crs["ukcp18"]
    elif "ukcordex" in fnm:
        return crs[run_name(re.sub(".+_", "", fnm))]
    

def lon180(ds, lon = "lon", lat = "lat"):
    if ds[lon].max() > 180:
        ds[lon] = (ds[lon].dims, (((ds[lon].values + 180) % 360) - 180), ds[lon].attrs)
        ds = ds.reindex({ lon : np.sort(ds[lon]) })
    ds = ds.reindex({ lat : np.sort(ds[lat]) })
    return ds


def transform_xy(x, y, crs_in = None, crs_out = None):
    
    # transform 1d or 2d coords into 2d coords in another projection
    
    x = np.array(x)
    y = np.array(y)
    
    if x.ndim == 1:
        # tile 1d coordinates into 2d arrays of x, y per cell
        x_2d = np.tile(np.array(x), (len(y), 1))
        y_2d = np.tile(np.array([y]).transpose(), (1, len(x)))
    else:
        # coordinates are already 2d arrays
        x_2d = x
        y_2d = y
    
    if crs_in is None or crs_out is None:
        # return tiled points on original grid
        return x_2d, y_2d
        
    else:
        # transform and return
        xy_transf = crs_out.transform_points(crs_in, x_2d, y_2d)
        return xy_transf[:,:,0], xy_transf[:,:,1]

    

def vertices_from_centres(x_coords, y_coords, crs_in = None, crs_out = None):
    
    # Compute vertices on regular grid of 1d x, y coordinates
    
    if np.array(x_coords).ndim > 1 or np.array(y_coords).ndim > 1:
        print("Coordinates must be one-dimensional")
        return
    
    # compute offsets needed to find midpoints
    x_offset = np.diff(x_coords) / 2
    y_offset = np.diff(y_coords) / 2
    
    # use offsets to obtain vectors of cell midpoints (adds 1 to each dimension)
    x_bounds = xr.concat([x_coords[0] - x_offset[0], x_coords[:-1] + x_offset, x_coords[-1] + x_offset[-1]], dim = x_coords.dims[0])
    y_bounds = xr.concat([y_coords[0] - y_offset[0], y_coords[:-1] + y_offset, y_coords[-1] + y_offset[-1]], dim = y_coords.dims[0])
    
    # expand vectors of bounds to grid of corners (& transform projection if necessary)
    xvert, yvert = transform_xy(x_bounds, y_bounds, crs_in = crs_in, crs_out = crs_out)
    
    return(xvert, yvert)


def px_and_bounds(x_coords, y_coords, crs_in = None, crs_out = None):
    
    lon, lat = transform_xy(x_coords, y_coords, crs_in = crs_in, crs_out = crs_out)
    lon_b, lat_b = vertices_from_centres(x_coords, y_coords, crs_in = crs_in, crs_out = crs_out)
    
    return {"lon" : lon, "lat" : lat, "lon_b" : lon_b, "lat_b" : lat_b}



def vertices_from_bounds(x_bounds, y_bounds, crs_in = None, crs_out = None):
    
    # Compute vertices from 2d bounds (where available, generally more accurate than computing from cell midpoints)
    if not (np.array(x_bounds).shape[1] == 2 and np.array(y_bounds).shape[1] == 2 and np.array(x_bounds).ndim == 2 and np.array(y_bounds).ndim == 2):
        print("Bounds DataArrays must each contain 2 vectors of coordinates")
        return
    
    # condense 2d bounds into 1d vector of midpoints
    x_midpoints = xr.concat([x_bounds[:,0], x_bounds[-1,1]], dim = x_bounds.dims[0])
    y_midpoints = xr.concat([y_bounds[:,0], y_bounds[-1,1]], dim = y_bounds.dims[0])

    xvert, yvert = transform_xy(x_midpoints, y_midpoints, crs_in = crs_in, crs_out = crs_out)

    return(xvert, yvert)



def regrid_to_osgb(da, from_mask, to_mask):
    
    # apply normed conservative regridders for land & sea regions to obtain a 'sharpened' coastline
    
    # both native & target grids must contain a field 'mask' consisting of ones and zeros
    # and must also contain coordinates 'lat' and 'lon' and vertices 'lat_b' and 'lon_b' 
    # defining cell centres and vertices on the same lat-lon grid (which may be regular or rotated)
    
    # regrid onto the attached mask
    regrid_ones = xe.Regridder(from_mask, to_mask, 'conservative_normed', unmapped_to_nan = True)
    rg_masked = regrid_ones(da).compute()
    
    # invert the mask & regrid again
    to_mask["mask"] = 1 - to_mask["mask"]
    from_mask["mask"] = 1 - from_mask["mask"]
    regrid_zeros = xe.Regridder(from_mask, to_mask, 'conservative_normed', unmapped_to_nan = True)
    rg_inv = regrid_zeros(da).compute()
    
    # add the two layers together
    rg_added = xr.concat([rg_masked, rg_inv], dim = "layer").sum(dim = "layer", skipna = True)
    
    # sort out the metadata
    rg_added = rg_added.assign_attrs(da.attrs)
    rg_added = rg_added.assign_coords(projection_y_coordinate = ("projection_y_coordinate", to_mask.projection_y_coordinate.data),
                                      projection_x_coordinate = ("projection_x_coordinate", to_mask.projection_x_coordinate.data))
    rg_added = rg_added.rename(lat = "grid_latitude", lon = "grid_longitude")
    
    return rg_added



def vertices_to_mesh(vlon, vlat):
    
    # convert 3d arrays containing vertices per point into 2d mesh of vertices
    
    # get dimensions to concatenate bounds over
    y = vlon.dims[0]
    x = vlon.dims[1]
    
    # concatenate vertices into all left-hand bounds & all right-hand bounds
    lh_lon, lh_lat = xr.concat([vlon[:,:,0], vlon[-1,:,3]], dim = y), xr.concat([vlat[:,:,0], vlat[-1,:,3]], dim = y)
    rh_lon, rh_lat = xr.concat([vlon[:,:,1], vlon[-1,:,2]], dim = y), xr.concat([vlat[:,:,1], vlat[-1,:,2]], dim = y)
    
    # now concatenate those into upper & lower bounds
    vert_lon = xr.concat([lh_lon, rh_lon[:,-1]], dim = x)
    vert_lat = xr.concat([lh_lat, rh_lat[:,-1]], dim = x)
    
    return vert_lon, vert_lat  



def rgcon_masked(ds, regrid_to):
    
    # method to carry out conservative regridding and remove any cells that are not fully covered by the source data.
    # Used when upscaling to a larger domain, to remove any possible edge effects.
    if "latitude" in ds.coords:
        ds = ds.rename(latitude = "lat", longitude = "lon")
    
    # create initial regridder
    rg_init = xe.Regridder(ds, regrid_to, 'conservative', unmapped_to_nan = True)
    
    # create mask to remove areas with less than 100% coverage (removes any cells with possible edge effects)
    ones = xr.DataArray(np.ones(ds.lat.shape), coords = ds.lat.coords)
    ttl_area = rg_init(ones).round(2)
    
    # add mask to target dataset
    regrid_to["mask"] = ttl_area.where(ttl_area == 1, 0)
    
    # construct regridder with output mask
    rgcon = xe.Regridder(ds, regrid_to, 'conservative', unmapped_to_nan = True)

    return rgcon



def combine_regionmasks(shapefiles, da, lon, lat, lon_name = "lon", lat_name = "lat", xdim = None, ydim = None,  aggregate_by = "mean"):
    
    # method to apply a list of regionmasks & concatenate the results into a single dataarray
    
    if xdim is None: xdim = lon_name
    if ydim is None: ydim = lat_name
    
    all_regions = []
    for sf in shapefiles:
        
        mask = regionmask.mask_3D_geopandas(sf, lon, lat, lon_name = lon_name, lat_name = lat_name, drop = False, numbers = "region")
        regions = getattr(da.where(mask), aggregate_by)([xdim, ydim])
        
        all_regions.append(regions)
        
    all_regions = xr.concat(all_regions, dim = "region")
    all_regions = all_regions.reindex(region = sorted(all_regions.region))
    all_regions = all_regions.assign_coords(geo_region = ("region", pd.concat(shapefiles).sort_values("region").geo_region.to_list()))
        
    return all_regions



def build_grid(x_coords, y_coords, crs_in = None, crs_out = None):
    
    # return grid with vertices based on regular grid of 1d x, y coordinates, to be used in regridding using xe.Regridder
    
    if np.array(x_coords).ndim > 1 or np.array(y_coords).ndim > 1:
        print("Coordinates must be one-dimensional")
        return
    
    lon, lat = transform_xy(x_coords, y_coords, crs_in, crs_out)
    lon_b, lat_b = vertices_from_centres(x_coords, y_coords, crs_in, crs_out)
    
    grid = {'lon': lon, 'lat': lat, 'lon_b': lon_b, 'lat_b': lat_b}
    
    return grid


def regridder(from_x, from_y, from_crs, to_x, to_y, to_crs, method, **kwargs):
    
    from_grid = build_grid(from_x, from_y, from_crs, to_crs)
    to_grid = build_grid(to_x, to_y)
    
    return xe.Regridder(from_grid, to_grid, method = method, **kwargs)




def add_grid(da, x_coords, y_coords, crs_in = None, crs_out = None):
    
    # return grid with vertices based on regular grid of 1d x, y coordinates, to be used in regridding using xe.Regridder
    if type(x_coords) == str: x_coords = da[x_coords]
    if type(y_coords) == str: y_coords = da[y_coords]
    
    if np.array(x_coords).ndim > 1 or np.array(y_coords).ndim > 1:
        print("Coordinates must be one-dimensional")
        return
    
    lon, lat = transform_xy(x_coords, y_coords, crs_in, crs_out)
    lon_b, lat_b = vertices_from_centres(x_coords, y_coords, crs_in, crs_out)
    
    da = da.assign_coords(lon = ([y_coords.name, x_coords.name], lon, x_coords.attrs), lat = ([y_coords.name, x_coords.name], lat, y_coords.attrs),
                          lon_b = ([y_coords.name+"_b", x_coords.name+"_b"], lon_b), lat_b = ([y_coords.name+"_b", x_coords.name+"_b"], lat_b))
    
    return da



def vec2map(x, mask):
    
    # reconstruct vector into map
    # create an empty map with NA in same cells as masks
    arr = mask.where(np.isnan(mask), 0)
    
    # get coordinates of non-empty cells
    px = np.argwhere(~np.isnan(mask.values))
    
    # Transfer vector values into non-empty cells in array
    if len(px) == len(x):
        for i in list(range(len(px))): arr[px[i,0], px[i,1]] = x[i]
        return arr
    else:
        print(str(len(x))+" values, but "+str(len(px))+" cells")
        return
    
    
def M2map(x, mask, labels = None):
    
    if labels is None:
        labels = {"n" : list(range(x.shape[0]))}
        
    # reconstruct matrix into map array
    # create an empty map with NA in same cells as mask
    arr = mask.where(np.isnan(mask), 0).expand_dims(labels).copy()
    
    # get coordinates of non-empty cells
    px = np.argwhere(~np.isnan(mask.values))
            
    if len(px) == x.shape[1]:
        for i in list(range(len(px))): arr[:, px[i,0], px[i,1]] = x[:,i]
        return arr
    else:
        print(str(len(x))+" values, but "+str(len(px))+" cells")
        return
    
    
def map2vec(da):
    
    # flatten 2d array into vector
    x = da.stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any")
    return x



def uk_lsm(lsm):
    
    # method to remove France & those odd bits of Scandinavia from the land-sea mask, by clustering adjacent points & removing any clusters that touch the right-hand boundary
    
    # identify X & Y axes, flatten into vectors of coords that DBscan can interpret
    axs = lsm.cf.axes
    x, y = transform_xy(lsm[axs["X"][0]], lsm[axs["Y"][0]])
    land_px = [[lon, lat] for lon, lat, lsm in zip(x.flatten(), y.flatten(), lsm.values.flatten()) if lsm == 1]
    
    # set cluster density based on grid resolution, and cluster the points
    eps = np.diff(lsm[axs["X"][0]]).mean() * 1.25
    clusts = DBSCAN(eps = eps, min_samples = 1).fit(land_px).labels_
    
    # remap vector of clusters onto LSM and identify clusters in rightmost column, which are to be removed
    rhs = np.unique(vec2map(clusts, lsm.where(lsm == 1)).values[:,-1])
        
    # flag those clusters as non-land and reshape into the map again
    lsm_uk = vec2map([not c in rhs for c in clusts.tolist()], lsm.where(lsm == 1))
    
    # finally, reset map to 0,1
    lsm_uk = lsm_uk.where(lsm_uk == 1, 0)
    
    return lsm_uk



def osgb_coords(city):
    
    location = Nominatim(user_agent="GetLoc")
    getLocation = location.geocode(city)
    px = crs["osgb"].transform_points(crs["latlon"], np.array(getLocation.longitude), np.array(getLocation.latitude))[:,:2][0]
    
    return px





def aggregate_by_shapefile(da, rname, lsm = None, da_crs = crs["rotated_latlon"]):
    
    # Aggregation over the native grid with a land-surface mask applied produces the closest maps to the data on CEDA, so that is the approach used here. 
    # Where the Channel Islands are omitted as a result, they are added back in from a single point on the coast of Jersey 
    # (see 00_notebooks/rg_aggregation_by_shapefile.ipynb for details of why this decision was made)
    
    # if passed a dataset, identify all variables of interest (these have X, Y and time dimensions) & drop the rest
    if type(da) == xr.core.dataset.Dataset:
        varnms = [v for v in da.data_vars if all([d in da[v].dims for d in list(da.reset_coords().coords)])]
        da = da[varnms]

    # drop ensemble_member dimension, if it exists
    da = da.squeeze(drop = True)
    
    # 'country' regions are split across three overlapping shapefiles that must be processed separately
    sf_list = [geopandas.read_file(fnm).to_crs(da_crs.proj4_init) for fnm in glob.glob("/data/met/processing/01_maps/ukcp18-uk-land-*"+rname+"*-hires")]

    regions = []
    for sf in sf_list[:1]:
        
        # create regionmasks - retain all regions, even if no cells are covered
        if "rlat" in da.coords:
            rmask = regionmask.mask_3D_geopandas(sf, da.rlon, da.rlat, lon_name = "rlon", lat_name = "rlat", drop = False, numbers = "region")
            collapse_dims = ["rlat", "rlon"]
        elif "x" in da.coords:
            da = da.assign_coords(lon = lsm.lon, lat = lsm.lat)
            rmask = regionmask.mask_3D_geopandas(sf, da.lon, da.lat, lon_name = "lon", lat_name = "lat", drop = False, numbers = "region")
            collapse_dims = ["x", "y"]
        else:
            print("Coords contain neither rlat/rlon nor x/y. Update needed to method.")
#             return
        
        # apply land surface mask if provided
        if lsm is not None:
            rmask = rmask.where(lsm.values == 1)
            

        # aggregate one region at a time to avoid crashing due to size of array produced (slower, but more stable)
        # also assign geo_region coordinate to ensure correct labelling
        region_avg = []
        for r in rmask.region.values:
            
            # extract region of interest & average over spatial dimension
            ra = da.where(rmask.sel(region = r)).mean(collapse_dims).reset_coords(drop = True)
            
            # assign region labels
            ra = ra.expand_dims(region = [r]).assign_coords(geo_region = ("region", [sf[sf.region == r]["geo_region"].values[0]]))
            region_avg.append(ra)
        region_avg = xr.concat(region_avg, "region")
        
        # if time dimension exists, reorder dims to ensure that time is first (needed for assigning Channel Island values)
        if "time" in region_avg.dims: region_avg = region_avg.transpose("time", "region", ...)

        regions.append(region_avg)
    regions = xr.concat(regions, "region").sortby("region")
        
        
    return regions



def land_regridder(fpath, wind = False, mask_only = False):
    
    # quickly build regridders for any model
    
    if "wind" in fpath or "Wind" in fpath or "wsgs" in fpath:
        wind = True
        
    # special cases may include wind fields, which are indexed by cell edges rather than vertices: will need to add these as they arise
        
    if "60km" in fpath:
        if wind:
            natgb = xr.open_dataset("/data/met/ukcp18/60km/fx/regridding-template-wind_natgb_ukcp18-60km.nc")
        else:
            natgb = xr.open_dataset("/data/met/ukcp18/60km/fx/regridding-template_natgb_ukcp18-60km.nc")
        osgb = xr.open_dataset("/data/met/ukcp18/60km/fx/regridding-template_osgb_ukcp18-60km.nc")
    
    elif "ukcp18" in fpath:
        if wind:
            natgb = xr.open_dataset("/data/met/ukcp18/fx/regridding-template-wind_natgb_ukcp18.nc").rename(lsm = "mask")
        else:
            natgb = xr.open_dataset("/data/met/ukcp18/fx/regridding-template_natgb_ukcp18.nc").rename(lsm = "mask")
        osgb = xr.open_dataset("/data/met/ukcp18/fx/regridding-template_osgb_ukcp18.nc").rename(lsm = "mask")
        
    elif "cmip5" in fpath:
        path_root = "/".join(fpath.split("/")[:6])
        if wind:
            natgb = xr.open_mfdataset(path_root+"/fx/regridding-template-wind_natgb_*.nc")
        else:
            natgb = xr.open_mfdataset(path_root+"/fx/regridding-template_natgb_*.nc")
        osgb = xr.open_mfdataset(path_root+"/fx/regridding-template_osgb_*.nc")
        
    elif "ukcordex" in fpath:
        path_root = "/".join(fpath.split("/")[:7])
        if wind:
            natgb = xr.open_mfdataset(path_root+"/fx/regridding-template-wind_natgb_*.nc")
        else:
            natgb = xr.open_mfdataset(path_root+"/fx/regridding-template_natgb_*.nc")
        osgb = xr.open_mfdataset(path_root+"/fx/regridding-template_osgb_*.nc")
        
    elif "ECMWF" in fpath:
        # same grid for wind & non-wind variables
        natgb = xr.open_mfdataset("/data/met/ECMWF-ERAINT/fx/regridding-template_natgb_ECMWF-ERAINT.nc")
        osgb = xr.open_mfdataset("/data/met/ECMWF-ERAINT/fx/regridding-template_osgb_ECMWF-ERAINT.nc")
        
    if mask_only: 
        return natgb
    else:
        regridder = xe.Regridder(natgb, osgb, "conservative_normed", unmapped_to_nan = True)
        return regridder



# quickly build on-the-fly regridder (no land-sea masking)

def qregridder(ds, src_crs = crs["rotated_latlon"]):
    
    src_grid = add_grid(ds.reset_coords(drop = True), ds.cf["X"], ds.cf["Y"], crs_in = src_crs, crs_out = crs["rotated_latlon"])
    
    k12 = xr.open_dataset("/data/met/ukcp18/fx/regridding-template_osgb_ukcp18.nc").reset_coords(drop = True)
    ukcp_grid = add_grid(k12.reset_coords(drop = True), k12.cf["X"], k12.cf["Y"], crs_in = crs["osgb"], crs_out = crs["rotated_latlon"])
    
    rg = xe.Regridder(src_grid, ukcp_grid, "conservative_normed", unmapped_to_nan = True)
    
    return rg



def grid(da, from_crs = None, to_crs = None):
    
    if to_crs is None: to_crs = from_crs
    # return grid with vertices based on regular grid of 1d x, y coordinates, to be used in regridding using xe.Regridder
        
    lon, lat = transform_xy(da.cf["X"], da.cf["Y"], from_crs, to_crs)
    lon_b, lat_b = vertices_from_centres(da.cf["X"], da.cf["Y"], from_crs, to_crs)
    
    grid = {'lon': lon, 'lat': lat, 'lon_b': lon_b, 'lat_b': lat_b}
    
    return grid