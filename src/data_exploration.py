# modified from: https://github.com/hdrake/cmip6-temperature-demo/blob/master/notebooks/00_calculate_simulated_global_warming.ipynb

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import cartopy
from tqdm.autonotebook import tqdm  # Fancy progress bars for our loops!
import intake
# util.py is in the local directory
# it contains code that is common across project notebooks
# or routines that are too extensive and might otherwise clutter
# the notebook design
# import util

plt.rcParams['figure.figsize'] = 12, 6


df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df.head()

col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

cat = col.search(experiment_id='historical',  # pick the `historical` forcing experiment
                 table_id='Amon',             # choose to look at atmospheric variables (A) saved at monthly resolution (mon)
                 variable_id='tas',           # choose to look at near-surface air temperature (tas) as our variable
                 member_id = 'r1i1p1f1')      # arbitrarily pick one realization for each model (i.e. just one set of initial conditions)

print('here')

time_slice = slice('1850','2019') # specific years that bracket our period of interest

# convert data catalog into a dictionary of xarray datasets
dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': False})

ds_dict = {}
gmst_dict = {}
i = 0
for name, ds in tqdm(dset_dict.items()):

    print(name)
    # continue
    # TODO: gotta debug this one
    if name == 'CMIP.MPI-M.ICON-ESM-LR.historical.Amon.gn':
        continue
    if name == 'CMIP.MPI-M.MPI-ESM1-2-HR.historical.Amon.gn':
        continue

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

    temperature_change = (
        ds['tas'].sel(time=slice('1990','2010')).mean(dim='time') -
        ds['tas'].sel(time=slice('1890','1910')).mean(dim='time')
    ).compute()
    temperature_change.attrs.update(ds.attrs)
    temperature_change = temperature_change.rename(
        r'temperature change ($^{\circ}$C) from 1890-1910 to 1990-2010'
    )

    import cartopy.crs as ccrs
    ortho = ccrs.Orthographic(-90, 20) # define target coordinate frame
    geo = ccrs.PlateCarree() # define origin coordinate frame

    plt.figure(figsize=(9,7))
    ax = plt.subplot(1, 1, 1, projection=ortho)

    q = temperature_change.plot(ax=ax, transform=geo, vmin=-6, vmax=6, cmap=plt.get_cmap('coolwarm')) # plot a colormap in transformed coordinates

    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    plt.title(f'Patterns of global warming over the Americas',fontsize=16, ha='center')
    plt.savefig(f'figs/historical_warming_patterns_{i}.png',dpi=100,bbox_inches='tight')
    plt.cla()
    plt.close('all')

    print('here')
    i += 1