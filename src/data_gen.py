# modified from: https://github.com/hdrake/cmip6-temperature-demo/blob/master/notebooks/00_calculate_simulated_global_warming.ipynb

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import cartopy
from tqdm.autonotebook import tqdm
import intake
import json

plt.rcParams['figure.figsize'] = 12, 6


df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df.head()

col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

cat = col.search(experiment_id='historical',  # pick the `historical` forcing experiment
                 table_id='Amon',             # choose to look at atmospheric variables (A) saved at monthly resolution (mon)
                 variable_id='tas',           # choose to look at near-surface air temperature (tas) as our variable
                 member_id = 'r1i1p1f1')      # arbitrarily pick one realization for each model (i.e. just one set of initial conditions)

print('here')

time_slice = slice('1850','2014') # specific years that bracket our period of interest

# convert data catalog into a dictionary of xarray datasets
dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': False})

ds_dict = {}
gmst_dict = {}
i = 0
name = 'CMIP.NASA-GISS.GISS-E2-1-G-CC.historical.Amon.gn'
ds = dset_dict[name]

print(name)

# rename spatial dimensions if necessary
if ('longitude' in ds.dims) or ('longitude' in ds):
    ds = ds.rename({'longitude':'lon'}) # some models labelled dimensions differently...
if ('latitude' in ds.dims) or ('latitude' in ds.dims):
    ds = ds.rename({'latitude':'lat'}) # some models labelled dimensions differently...

ds = xr.decode_cf(ds) # temporary hack, not sure why I need this but has to do with calendar-aware metadata on the time variable
ds = ds.sel(time=time_slice) # subset the data for the time period of interest

# drop redundant variables (like "height: 2m")
for coord in ds.coords:
    if coord not in ['lat','lon','time']:
        ds = ds.drop(coord)

## Calculate global-mean surface temperature (GMST)
cos_lat_2d = np.cos(np.deg2rad(ds['lat'])) * xr.ones_like(ds['lon']) # effective area weights
gmst = (
    (ds['tas'] * cos_lat_2d).sum(dim=['lat','lon']) /
    cos_lat_2d.sum(dim=['lat','lon'])
)

# Add GMST to dictionary
gmst_dict[name] = gmst.squeeze()

# Add near-surface air temperature to dictionary
ds_dict[name] = ds

train_data = []
lats = ds['tas'].sel(time=slice('1990','2010')).mean(dim='time').lat.values
lons = ds['tas'].sel(time=slice('1990','2010')).mean(dim='time').lon.values

test_decades = (1920, 2010)

test_data_space = []
test_data_time = []
test_data_space_time = []

for i in range(1850, 2020, 10):
    temps = ds['tas'].sel(time=slice(str(i), str(i+10))).mean(dim='time').values.squeeze()
    decade = i

    for lat_idx in range(len(lats)):
        for lon_idx in range(len(lons)):

            valid_space = lat_idx % 2 == 1 or lon_idx % 2 == 1
            valid_time = i not in test_decades

            curr_dataset = None

            if valid_space and valid_time:
                curr_dataset = train_data
            elif valid_space and not valid_time:
                curr_dataset = test_data_time
            elif not valid_space and valid_time:
                curr_dataset = test_data_space
            else:
                curr_dataset = test_data_space_time

            curr_dataset.append(
                {
                    "lat": lats[lat_idx].item(),
                    "long": lons[lon_idx].item(),
                    "time": i,
                    "temp": temps[lat_idx, lon_idx].item()
                }
            )

with open('/local/crv/sagadre/repos/wilds-fine-tuning/src/data/curr/train.json', 'w') as f:
    json.dump(train_data, f, indent=4)
with open('/local/crv/sagadre/repos/wilds-fine-tuning/src/data/curr/test_time.json', 'w') as f:
    json.dump(test_data_time, f, indent=4)
with open('/local/crv/sagadre/repos/wilds-fine-tuning/src/data/curr/test_space.json', 'w') as f:
    json.dump(test_data_space, f, indent=4)
with open('/local/crv/sagadre/repos/wilds-fine-tuning/src/data/curr/test_space_time.json', 'w') as f:
    json.dump(test_data_space_time, f, indent=4)
