import numpy as np
import xarray as xr
from scipy import fftpack


# Methods to compute jetstream indices, based on Woollings2010 (code developed by Thomas Keel)


### Data formatting
def swap_coord_order(data, coord, ascending=True):
    """
        Will reverse the dimension if a higher number is first
        
        Parameters
        ----------
        data : xarray.Dataset
            climate data
        coord : str
            name from coord to change

        Useage
        ----------
        new_data = swap_coord_order(data, "lat")
    """
    first_val = 0
    last_val = -1
    if not ascending:
        first_val = -1
        last_val = 0
    if data[coord][first_val] > data[coord][last_val]:
        data = data.reindex(**{coord:list(reversed(data[coord]))})
    return data


def subset_data(data):
    for coord in ['plev', 'lat', 'lon']:
        assert coord in data.coords
        data = swap_coord_order(data, coord)
    return data.sel(plev=slice(70000, 92500), lat=slice(15, 75), lon=slice(120, 180))


### Metric
def run_metric(data, filter_freq=10, window_size=61):
    """
        Follows an in-text description of 4-steps describing the algorithm mof jet-stream identification from Woolings et al. (2010). 
        Will calculate this metric based on data (regardless of pressure level of time span etc.). 
        
        Parameters
        ----------
        data (xarray.Dataset): input data containing u and v wind
        filter_freq (int): number of days in filter
        window_size (int): number of days in window for Lancoz filter

        returns:
            max_lat_ws (numpy.ndarray):
    """
    ## Step 1
    print('Step 1: calculating long and/or plev mean...')
    mean_data = get_zonal_mean(data)
    ## Step 2
    print('Step 2: Applying %s day lanczos filter...' % (filter_freq))
    lancoz_filtered_mean_data = apply_lanczos_filter(mean_data, filter_freq, window_size)
    ## Step 3
    print('Step 3: Calculating max windspeed and latitude where max windspeed found...')
    max_lat_ws = np.array(list(map(get_latitude_and_speed_where_max_ws, lancoz_filtered_mean_data[:])))
    mean_data_lat_ws = assign_lat_ws_to_data(mean_data, max_lat_ws)
    ## Step 4
    print('Step 4: Make climatology')
    climatology = make_climatology(mean_data_lat_ws, 'month')
    ## Step 5
    print('Step 5: Apply low-freq fourier filter to both max lats and max windspeed')
    fourier_filtered_lats = apply_low_freq_fourier_filter(climatology['max_lats'].values, highest_freq_to_keep=2)
    fourier_filtered_ws = apply_low_freq_fourier_filter(climatology['max_ws'].values, highest_freq_to_keep=2)
    ## Step 6
    print('Step 6: Join filtered climatology back to the data')
    time_dim = climatology['max_ws'].dims[0]
    fourier_filtered_data = assign_filtered_vals_to_data(mean_data_lat_ws, fourier_filtered_lats, fourier_filtered_ws, dim=time_dim)
    return fourier_filtered_data
    

def make_climatology(data, freq):
    """
        Makes a climatology at given interval (i.e. days, months, season)
        
        Parameters
        ----------
        data (xarray.Dataset): data with regular time stamp
        freq (str): 'day', 'month' or 'season'
        
        Usage
        ----------
        climatology = make_climatology(data, 'month')
        
        
    """
    climatology = data.groupby("time.%s" % (freq)).mean("time")
    return climatology


def get_zonal_mean(data):
    """
        Will get the zonal mean either by pressure level (plev) or for one layer
        Used in Woolings et al. 2010
    """
    if not 'lon' in data.coords:
        raise KeyError("data does not contain 'lon' coord")
        
    coords_for_mean = ['lon', 'plev']
    if 'plev' not in data.coords:
        coords_for_mean = ['lon']
    mean_data = data.mean(coords_for_mean)
    return mean_data


def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.
    
    A low-pass filter removes short-term random fluctations in a time series

    Used in Woolings et al. 2010

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    TAKEN FROM: https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[0+(window%2):-1] # edited from w[1:-1]


def apply_lanczos_filter(data, filter_freq, window_size):
    """
        Will carry out Lanczos low-pass filter

        Used in Woolings et al. 2010
    """
    assert filter_freq <= data['time'].count() and filter_freq > 0, "Filter frequency needs to be less\
                                                                     than the number of days in the data\
                                                                    and more than 0 "
    assert window_size <= data['time'].count() and window_size > 0, "Window size needs to be less\
                                                                     than the number of days in the data\
                                                                     and more than 0 "
    assert filter_freq <= window_size, "Filter freq cannot be bigger than window size"

    lanczos_weights = low_pass_weights(window_size, 1/filter_freq)
    lanczos_weights_arr = xr.DataArray(lanczos_weights, dims=['window'])
    window_cons = data['ua'].rolling(time=len(lanczos_weights_arr), center=True).construct('window').dot(lanczos_weights_arr)
    return window_cons


def get_latitude_and_speed_where_max_ws(data_row):
    """
        Will return the latitude and windspeed at the index of maximum wind speed from a row of data
        Used in Woolings et al. 2010
    """
    try:
        assert hasattr(data_row, 'isnull')
    except:
        raise AttributeError("input needs to have isnull method")
    
    if not data_row.isnull().all():
        data_row = data_row.fillna(0.0)
        max_speed_loc = np.argmax(data_row.data)
        max_speed = data_row[max_speed_loc]
        lat_at_max = float(max_speed['lat'].values)
        speed_at_max = float(max_speed.data)
        return lat_at_max, speed_at_max 
    else:
        return None, None


def assign_lat_ws_to_data(data, max_lat_ws):
    """
        Will return a data array with the maximum windspeed and latitude of that 
        maximum wind speed
        Used in Woolings et al. 2010
    """
    max_lats = max_lat_ws[:,0]
    max_ws = max_lat_ws[:,1]
    data_with_max_lats_ws = data.assign({'max_lats':(('time'),max_lats), 'max_ws':(('time'),max_ws)})
    data_with_max_lats_ws['max_lats'] = data_with_max_lats_ws['max_lats'].astype(float)
    data_with_max_lats_ws['max_ws'] = data_with_max_lats_ws['max_ws'].astype(float)
    return data_with_max_lats_ws


def apply_low_freq_fourier_filter(data, highest_freq_to_keep):
    """
        Carries out a Fourier transform for filtering keeping only low frequencies
        ADAPTED FROM: https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
        
        Used in Woolings et al. 2010
        Parameters
        ----------
        data : (np.array - 1-d) 
            time series data at regular intervals
        highest_freq_to_keep : (int)
            highest frequency to keep in the fourier transform expression
            NOTE: starts at 0, so highest_freq_to_keep=1 will only keep the constant and first expresion
            
        
        Usage
        ----------
        # Apply filter of the two lowest frequencies
        apply_low_freq_fourier_filter(data, highest_freq_to_keep=2)
            
    """
    ## Fast Fourier Transform on the time series data
    fourier_transform = fftpack.fft(data)
    
    ## Remove low frequencies
    fourier_transform[highest_freq_to_keep+1:] = 0
    
    ## Inverse Fast Fourier Transform the time series data back
    filtered_sig = fftpack.ifft(fourier_transform)
    return filtered_sig


def assign_filtered_vals_to_data(data, filtered_max_lats, filtered_max_ws, dim):
    """
        Assigns the filtered data back to the returned dataset
        Used in Woolings et al. 2010
        
    """
    filtered_data = data.assign({'ff_max_lats':((dim), filtered_max_lats),\
                      'ff_max_ws':((dim), filtered_max_ws)})
    filtered_data['ff_max_lats'] = filtered_data['ff_max_lats'].astype(float)
    filtered_data['ff_max_ws'] = filtered_data['ff_max_ws'].astype(float)
    return filtered_data
