
import pandas as pd
import numpy as np
import xarray as xr; xr.set_options(keep_attrs=True)
import xclim
import glob
import re
from datetime import timedelta, datetime

import sys
sys.path.append('/data/met/processing/10_methods/')
from regridding import *
from dicts import *
from misc import run_name, tabulate

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)          # silence annoying warnings about future changes in dask default behaviour
warnings.filterwarnings("ignore", message = ".+rlat.+")              # silence warnings that rlat & rlon are being used for the regridding - this is correct
warnings.filterwarnings("ignore", message = ".+time\.encoding.+")    # silence warnings that time bounds may not have correct encoding
warnings.filterwarnings("ignore", category = RuntimeWarning)         # silence warnings about division by zero/NaN when aggregating by region

import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})       # silence warning about creating large chunk

from IPython.display import clear_output

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Definition of transverse Mercator projection used for OSGB data

transverse_mercator = {'grid_mapping_name': 'transverse_mercator',
                       'longitude_of_prime_meridian': 0.0, 
                       'semi_major_axis': 6377563.396, 
                       'semi_minor_axis': 6356256.909, 
                       'longitude_of_central_meridian': -2.0,
                       'latitude_of_projection_origin': 49.0, 
                       'false_easting': 400000.0,
                       'false_northing': -100000.0,
                       'scale_factor_at_central_meridian': 0.9996012717}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Attributes

attrs = {'CNRM-CERFACS-CNRM-CM5_r1i1p1_HadREM3-GA7-05': {'institution': 'MetOffice, Hadley Centre, UK',
  'institute_id': 'MOHC',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'MOHC-HadREM3-GA7-05',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.4',
  'references': 'https://www.metoffice.gov.uk/climate-guide/science/science-behind-climate-change/hadley; The Met Office Unified Model Global Atmosphere 7.0/7.1 and JULES Global Land 7.0 configurations Waters et al.'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_ALADIN63': {'institution': 'CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France)',
  'institute_id': 'CNRM',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CNRM-ALADIN63',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.6',
  'references': 'http://www.umr-cnrm.fr/spip.php?article125&lang=en'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/weather-climate-models'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.6', 'references': ''},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_REMO2015': {'institution': 'Helmholtz-Zentrum Geesthacht, Climate Service Center Germany',
  'institute_id': 'GERICS',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'GERICS-REMO2015',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.4',
  'references': 'http://www.remo-rcm.de'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_WRF381P': {'institution': 'Institute Pierre Simon Laplace',
  'institute_id': 'IPSL',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'IPSL-WRF381P',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.6',
  'references': 'https://cmc.ipsl.fr/'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_CCLM4-8-17': {'institution': 'Climate Limited-area Modelling Community (CLM-Community)',
  'institute_id': 'CLMcom',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-CCLM4-8-17',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.clm-community.eu/'},
 'CNRM-CERFACS-CNRM-CM5_r1i1p1_RegCM4-6': {'institution': 'International Centre for Theoretical Physics',
  'institute_id': 'ICTP',
  'driving_model_id': 'CNRM-CERFACS-CNRM-CM5',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'ICTP-RegCM4-6',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.7',
  'references': 'http://gforge.ictp.it/gf/project/regcm'},
 'MOHC-HadGEM2-ES_r1i1p1_HadREM3-GA7-05': {'institution': 'MetOffice, Hadley Centre, UK',
  'institute_id': 'MOHC',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'MOHC-HadREM3-GA7-05',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'https://www.metoffice.gov.uk/climate-guide/science/science-behind-climate-change/hadley; The Met Office Unified Model Global Atmosphere 7.0/7.1 and JULES Global Land 7.0 configurations Waters et al.'},
 'MOHC-HadGEM2-ES_r1i1p1_ALADIN63': {'institution': 'CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France)',
  'institute_id': 'CNRM',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CNRM-ALADIN63',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'http://www.umr-cnrm.fr/spip.php?article125&lang=en'},
 'MOHC-HadGEM2-ES_r1i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'MOHC-HadGEM2-ES_r1i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/regional_climate'},
 'MOHC-HadGEM2-ES_r1i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.6', 'references': ''},
 'MOHC-HadGEM2-ES_r1i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'MOHC-HadGEM2-ES_r1i1p1_RegCM4-6': {'institution': 'International Centre for Theoretical Physics',
  'institute_id': 'ICTP',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'ICTP-RegCM4-6',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://gforge.ictp.it/gf/project/regcm'},
 'MOHC-HadGEM2-ES_r1i1p1_WRF381P': {'institution': 'Institute Pierre Simon Laplace',
  'institute_id': 'IPSL',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'IPSL-WRF381P',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'https://cmc.ipsl.fr/'},
 'MOHC-HadGEM2-ES_r1i1p1_CCLM4-8-17': {'institution': 'Climate Limited-area Modelling Community (CLM-Community)',
  'institute_id': 'CLMcom',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-CCLM4-8-17',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.clm-community.eu/'},
 'MOHC-HadGEM2-ES_r1i1p1_REMO2015': {'institution': 'Helmholtz-Zentrum Geesthacht, Climate Service Center Germany',
  'institute_id': 'GERICS',
  'driving_model_id': 'MOHC-HadGEM2-ES',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'GERICS-REMO2015',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.remo-rcm.de/'},
 'ICHEC-EC-EARTH_r12i1p1_HadREM3-GA7-05': {'institution': 'MetOffice, Hadley Centre, UK',
  'institute_id': 'MOHC',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'MOHC-HadREM3-GA7-05',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'https://www.metoffice.gov.uk/climate-guide/science/science-behind-climate-change/hadley; The Met Office Unified Model Global Atmosphere 7.0/7.1 and JULES Global Land 7.0 configurations Waters et al.'},
 'ICHEC-EC-EARTH_r12i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'ICHEC-EC-EARTH_r1i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'ICHEC-EC-EARTH_r3i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'ICHEC-EC-EARTH_r12i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/regional_climate'},
 'ICHEC-EC-EARTH_r1i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/regional_climate'},
 'ICHEC-EC-EARTH_r3i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/weather-climate-models'},
 'ICHEC-EC-EARTH_r1i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6', 'references': ''},
 'ICHEC-EC-EARTH_r12i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6', 'references': ''},
 'ICHEC-EC-EARTH_r3i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v2',
  'Conventions': 'CF-1.6', 'references': ''},
 'ICHEC-EC-EARTH_r12i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'ICHEC-EC-EARTH_r1i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'ICHEC-EC-EARTH_r3i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'ICHEC-EC-EARTH_r12i1p1_RegCM4-6': {'institution': 'International Centre for Theoretical Physics',
  'institute_id': 'ICTP',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'ICTP-RegCM4-6',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.7',
  'references': 'http://gforge.ictp.it/gf/project/regcm'},
 'ICHEC-EC-EARTH_r12i1p1_WRF381P': {'institution': 'Institute Pierre Simon Laplace',
  'institute_id': 'IPSL',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'IPSL-WRF381P',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'https://cmc.ipsl.fr/'},
 'ICHEC-EC-EARTH_r12i1p1_CCLM4-8-17': {'institution': 'Climate Limited-area Modelling Community (CLM-Community)',
  'institute_id': 'CLMcom',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'CLMcom-CCLM4-8-17',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.clm-community.eu/'},
 'ICHEC-EC-EARTH_r12i1p1_REMO2015': {'institution': 'Helmholtz-Zentrum Geesthacht, Climate Service Center Germany',
  'institute_id': 'GERICS',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'GERICS-REMO2015',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.remo-rcm.de/'},
 'MPI-M-MPI-ESM-LR_r1i1p1_HadREM3-GA7-05': {'institution': 'MetOffice, Hadley Centre, UK',
  'institute_id': 'MOHC',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'MOHC-HadREM3-GA7-05',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'https://www.metoffice.gov.uk/climate-guide/science/science-behind-climate-change/hadley; The Met Office Unified Model Global Atmosphere 7.0/7.1 and JULES Global Land 7.0 configurations Waters et al.'},
 'MPI-M-MPI-ESM-LR_r1i1p1_ALADIN63': {'institution': 'CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France)',
  'institute_id': 'CNRM',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CNRM-ALADIN63',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'http://www.umr-cnrm.fr/spip.php?article125&lang=en'},
 'MPI-M-MPI-ESM-LR_r1i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'MPI-M-MPI-ESM-LR_r2i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r2i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'MPI-M-MPI-ESM-LR_r3i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'MPI-M-MPI-ESM-LR_r1i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/weather-climate-models'},
 'MPI-M-MPI-ESM-LR_r1i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6', 'references': ''},
 'MPI-M-MPI-ESM-LR_r1i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1a',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'MPI-M-MPI-ESM-LR_r2i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r2i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'MPI-M-MPI-ESM-LR_r3i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'MPI-M-MPI-ESM-LR_r3i1p1_REMO2015': {'institution': 'Helmholtz-Zentrum Geesthacht, Climate Service Center Germany',
  'institute_id': 'GERICS',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r3i1p1',
  'model_id': 'GERICS-REMO2015',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.remo-rcm.de'},
 'MPI-M-MPI-ESM-LR_r1i1p1_RegCM4-6': {'institution': 'International Centre for Theoretical Physics',
  'institute_id': 'ICTP',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'ICTP-RegCM4-6',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.7',
  'references': 'http://gforge.ictp.it/gf/project/regcm'},
 'MPI-M-MPI-ESM-LR_r1i1p1_CCLM4-8-17': {'institution': 'Climate Limited-area Modelling Community (CLM-Community)',
  'institute_id': 'CLMcom',
  'driving_model_id': 'MPI-M-MPI-ESM-LR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-CCLM4-8-17',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.clm-community.eu/'},
 'MPI-M-MPI-ESM-LR_r1i1p1_WRF381P': {'institution': 'Institute Pierre Simon Laplace',
  'institute_id': 'IPSL',
  'driving_model_id': 'ICHEC-EC-EARTH',
  'driving_model_ensemble_member': 'r12i1p1',
  'model_id': 'IPSL-WRF381P',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'https://cmc.ipsl.fr/'},
 'NCC-NorESM1-M_r1i1p1_HadREM3-GA7-05': {'institution': 'MetOffice, Hadley Centre, UK',
  'institute_id': 'MOHC',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'MOHC-HadREM3-GA7-05',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'https://www.metoffice.gov.uk/climate-guide/science/science-behind-climate-change/hadley; The Met Office Unified Model Global Atmosphere 7.0/7.1 and JULES Global Land 7.0 configurations Waters et al.'},
 'NCC-NorESM1-M_r1i1p1_ALADIN63': {'institution': 'CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France)',
  'institute_id': 'CNRM',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CNRM-ALADIN63',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'http://www.umr-cnrm.fr/spip.php?article125&lang=en'},
 'NCC-NorESM1-M_r1i1p1_COSMO-crCLIM-v1-1': {'institution': 'ETH Zurich, Zurich, Switzerland in collaboration with the CLM-Community',
  'institute_id': 'CLMcom-ETH',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'CLMcom-ETH-COSMO-crCLIM-v1-1',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://cordex.clm-community.eu/'},
 'NCC-NorESM1-M_r1i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/weather-climate-models'},
 'NCC-NorESM1-M_r1i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v3',
  'Conventions': 'CF-1.6', 'references': ''},
 'NCC-NorESM1-M_r1i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'NCC-NorESM1-M_r1i1p1_REMO2015': {'institution': 'Helmholtz-Zentrum Geesthacht, Climate Service Center Germany',
  'institute_id': 'GERICS',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'GERICS-REMO2015',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.remo-rcm.de/'},
 'NCC-NorESM1-M_r1i1p1_RegCM4-6': {'institution': 'International Centre for Theoretical Physics',
  'institute_id': 'ICTP',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'ICTP-RegCM4-6',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.7',
  'references': 'http://gforge.ictp.it/gf/project/regcm'},
 'NCC-NorESM1-M_r1i1p1_WRF381P': {'institution': 'Institute Pierre Simon Laplace',
  'institute_id': 'IPSL',
  'driving_model_id': 'NCC-NorESM1-M',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'IPSL-WRF381P',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'https://cmc.ipsl.fr/'},
 'IPSL-IPSL-CM5A-MR_r1i1p1_RACMO22E': {'institution': 'Royal Netherlands Meteorological Institute, De Bilt, The Netherlands',
  'institute_id': 'KNMI',
  'driving_model_id': 'IPSL-IPSL-CM5A-MR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'KNMI-RACMO22E',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.knmi.nl/research/weather-climate-models'},
 'IPSL-IPSL-CM5A-MR_r1i1p1_HIRHAM5': {'institution': 'Danish Meteorological Institute',
  'institute_id': 'DMI',
  'driving_model_id': 'IPSL-IPSL-CM5A-MR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'DMI-HIRHAM5',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6', 'references': ''},
 'IPSL-IPSL-CM5A-MR_r1i1p1_RCA4': {'institution': 'Swedish Meteorological and Hydrological Institute, Rossby Centre',
  'institute_id': 'SMHI',
  'driving_model_id': 'IPSL-IPSL-CM5A-MR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'SMHI-RCA4',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre'},
 'IPSL-IPSL-CM5A-MR_r1i1p1_REMO2015': {'institution': 'Helmholtz-Zentrum Geesthacht, Climate Service Center Germany',
  'institute_id': 'GERICS',
  'driving_model_id': 'IPSL-IPSL-CM5A-MR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'GERICS-REMO2015',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.4',
  'references': 'http://www.remo-rcm.de'},
 'IPSL-IPSL-CM5A-MR_r1i1p1_WRF381P': {'institution': 'Institute Pierre Simon Laplace',
  'institute_id': 'IPSL',
  'driving_model_id': 'IPSL-IPSL-CM5A-MR',
  'driving_model_ensemble_member': 'r1i1p1',
  'model_id': 'IPSL-WRF381P',
  'rcm_version_id': 'v1',
  'Conventions': 'CF-1.6',
  'references': 'https://cmc.ipsl.fr/'}}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# and some handy functions

def ceda_runs(sres = False, tres = False):
    
    # count number of runs available in CEDA directories
    
    if len(glob.glob("/data/met/ukcordex/*/*/*/ceda/*.nc")) == 0:
        print("No CEDA runs currently available")
        return
    
    df = pd.DataFrame([x.split("/") for x in glob.glob("/data/met/ukcordex/*/*/*/ceda/*.nc")])[[4,5,6,8]]
    df["run"] = df[4]+"_"+df[5]+"_"+df[6]
    df[["id", "varnm", "sres", "tres",]] = df[8].str.split("_", expand = True)[[5,0,4,6]]
    
    cols = [df.varnm]
    if sres: cols = cols + [df.sres]
    if tres: cols = cols + [df.tres]
    
    return pd.crosstab(index=[df.id, df.run], columns=cols)
    
    
    
def aggregate_osgb(ds, freq = "MS"):
    
    # Aggregate CEDA-compliant daily data using 'resample'
    
    # identify & aggregate the main variables (will always have 'ensemble_member' dimension in this process)
    varnms = [v for v in ds.data_vars if "ensemble_member" in ds[v].dims]
    newdat = {v : (ds[v].dims, ds[v].resample(time = freq).mean().data, ds[v].attrs) for v in varnms}

    # fix metadata depending on specified frequency
    if freq == "MS":
        tmplt = xr.open_dataset("/data/met/processing/02_ceda_templates/tas_rcp85_land-rcm_uk_12km_01_mon_198012-208011.nc")
        extra_coords = { "month_number" : tmplt.month_number, "year" : tmplt.year, "yyyymm" : tmplt.yyyymm}
        freq_str = "mon"
        coord_order = "ensemble_member_id grid_latitude grid_longitude month_number year yyyymm"
    elif freq == "QS-DEC":
        tmplt = xr.open_dataset("/data/met/processing/02_ceda_templates/tas_rcp85_land-rcm_uk_12km_01_seas_198012-208011.nc")
        extra_coords = {"season" : tmplt.season, "year" : tmplt.year}
        freq_str = "seas"
        coord_order = "ensemble_member_id grid_latitude grid_longitude season year" 
    elif freq == "AS-DEC":
        tmplt = xr.open_dataset("/data/met/processing/02_ceda_templates/tas_rcp85_land-rcm_uk_12km_01_ann_198012-208011.nc")
        extra_coords = {"year" : tmplt.year}
        freq_str = "ann"
        coord_order = "ensemble_member_id grid_latitude grid_longitude year"
           
    # can now create the dataset:
    if "projection_x_coordinate" in ds.coords:
        newdat.update(transverse_mercator = ds.transverse_mercator,
                                 time_bnds = tmplt.time_bnds,
                                 projection_y_coordinate_bnds = tmplt.projection_y_coordinate_bnds,
                                 projection_x_coordinate_bnds = tmplt.projection_x_coordinate_bnds)
        
        ds_agg = xr.Dataset(data_vars = newdat,
                            coords = {"ensemble_member_id" : ds.ensemble_member_id,
                                      "grid_latitude" : ds.grid_latitude,
                                      "grid_longitude" : ds.grid_longitude
                                     },
                            attrs = ds.attrs).assign_attrs(frequency = freq_str).assign_coords(extra_coords)
    else:
        newdat.update(time_bnds = tmplt.time_bnds)
        coord_order = re.sub("grid_latitude grid_longitude", "geo_region", coord_order)
        
        ds_agg = xr.Dataset(data_vars = newdat,
                            coords = {"ensemble_member_id" : ds.ensemble_member_id,
                                      "geo_region" : ds.geo_region,
                                     },
                            attrs = ds.attrs).assign_attrs(frequency = freq_str).assign_coords(extra_coords)
        
    # encoding
    ds_agg = ds_agg.transpose("ensemble_member", "time", ...)
    ds_agg["ensemble_member"].encoding["dtype"] = "intc"
    
    for varnm in varnms:
        ds_agg[varnm].encoding["coordinates"] = coord_order
        ds_agg[varnm].encoding["dtype"] = "float32"
    
    return ds_agg



def aggregate_slices(ds, fpath, slice_length = 30, verbose = False):
    
    varnm = list(ds.data_vars)[0]

    fpath = fpath+"/ceda/"+varnm+"_rcp85_land-eurocordex_uk_"
           
    tmplt_fnms = glob.glob("/data/met/processing/02_ceda_templates/tas_rcp85_land-rcm_uk_12km_01_"+ds.frequency+"-"+str(slice_length)+"y_*.nc")
    
    # create separate files for historical & future periods, as per CEDA
    for fnm in tmplt_fnms:
        tmplt = xr.open_dataset(fnm).load()
                
        if "time" in tmplt.dims:
            slices = []
            for i in range(len(tmplt.time)):
                
                sl = ds.sel(time = slice(tmplt.time_bnds[i,0].dt.strftime("%Y%m%d").values.tolist(), tmplt.time_bnds[i,1].dt.strftime("%Y%m%d").values.tolist()))

                if ds.frequency == "mon":
                    sl = sl.sel(time = sl.month_number == tmplt.time[i].month_number)
                    coord_order = "ensemble_member_id grid_latitude grid_longitude month_number yyyymm"
                elif ds.frequency == "seas":
                    sl = sl.sel(time = sl.season == tmplt.time[i].season)
                    coord_order = "ensemble_member_id grid_latitude grid_longitude season year" 
                elif ds.frequency == "ann":
                    coord_order = "ensemble_member_id grid_latitude grid_longitude time year"
                    
                # check that right number of periods is included
                if len(sl.time) != slice_length:
                    print("**ERROR** Slice length is "+str(len(sl.time)))
                    return
                    
                slices.append(sl.mean("time"))
                
            slices = xr.concat(slices, "time").transpose("ensemble_member", "time", ...)
        
        else:
            # only occurs for annual frequency, historical period
            coord_order = "ensemble_member_id grid_latitude grid_longitude time year"
            sl = ds.sel(time = slice(tmplt.time_bnds[0].dt.strftime("%Y%m%d").values.tolist(), tmplt.time_bnds[1].dt.strftime("%Y%m%d").values.tolist()))
            slices = sl.mean("time")
        
        # make time a coordinate dimension again, reassign time bounds & reassign spatial coords to remove time dimension
        slices = slices.assign_coords(time = tmplt.time)
        slices["time_bnds"] = tmplt.time_bnds
        
        if "projection_y_coordinate" in ds.dims:
            fnm_root = fpath + "12km_"+str(ds.ensemble_member.values[0])
            slices["transverse_mercator"] = tmplt.transverse_mercator
            slices["projection_y_coordinate_bnds"] = tmplt.projection_y_coordinate_bnds
            slices["projection_x_coordinate_bnds"] = tmplt.projection_x_coordinate_bnds
        else:
            fnm_root = fpath + ds.resolution+"_"+str(ds.ensemble_member.values[0])
            coord_order = re.sub("grid_latitude grid_longitude", "geo_region", coord_order)
            
        slices[varnm].encoding["coordinates"] = coord_order
        slices[varnm].encoding["dtype"] = "float32"
        slices["ensemble_member"].encoding["dtype"] = "intc"
        
        new_fnm = fnm_root+"_"+ds.frequency+"-"+str(slice_length)+"y_"+re.sub(".+"+ds.frequency+"-"+str(slice_length)+"y_", "", fnm)
        slices.to_netcdf(new_fnm)
        if verbose: print("    ..." + new_fnm)