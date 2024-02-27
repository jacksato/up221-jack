import pandas as pd
import geopandas as gpd 
import plotly.express as px
import matplotlib.pyplot as plt
from itables import init_notebook_mode
from itables.sample_dfs import get_countries
import folium as fm
from folium.plugins import HeatMap
import numpy as np


evictions = pd.read_csv('/Users/jackfukushima/Python/data/eviction_data_concat.csv')
evictions_geocoded = pd.read_csv('/Users/jackfukushima/Python/data/evictions_geocoded.csv')

# Convert evictions 'Address' to string
evictions['Address'] = evictions['Address'].astype(str)

# Create new column in evictions_geocode with matching name 'Address' and make sure it's a string
evictions_geocoded['Address'] = evictions_geocoded['input_string'].astype(str)
evictions_matched = evictions_geocoded.merge(evictions,left_index=True,right_index=True)

evictions_matched.head()
evic_geo_match = gpd.GeoDataFrame(evictions_matched,geometry=gpd.points_from_xy(evictions_matched['longitude'],evictions_matched['latitude']),crs='4326')

svi_la = gpd.read_file('/Users/jackfukushima/Python/data/LACounty_SVI_gdb_-2756885108865704561.geojson')
svi_la = svi_la[[
    'geometry',
    'FIPS',
    'LOCATION',
    'AREA_SQMI',
    'E_TOTPOP',
    'E_HU',
    'E_HH',
    'E_POV',
    'E_UNEMP',
    'E_PCI',
    'E_NOHSDP',
    'E_AGE65',
    'E_AGE17',
    'E_DISABL',
    'E_SNGPNT',
    'E_MINRTY',
    'E_LIMENG',
    'E_MUNIT',
    'E_MOBILE',
    'E_CROWD',
    'E_NOVEH',
    'E_GROUPQ',
    'EP_POV',
    'EP_UNEMP',
    'EP_PCI',
]].copy()
svi_la = svi_la.rename(
    columns={
    'LOCATION':'location',
    'AREA_SQMI':'area_sqmile',
    'E_TOTPOP':'total_pop',
    'E_HU':'housing_units',
    'E_HH':'households',
    'E_POV':'pct_below150',
    'E_UNEMP':'unemployed',
    'E_PCI':'cost_burdened_low_income',
    'E_NOHSDP':'no_high_school',
    'E_AGE65':'persons_over65',
    'E_AGE17':'persons_under17',
    'E_DISABL':'disabled_pop',
    'E_SNGPNT':'single_parent',
    'E_MINRTY':'minority_pop',
    'E_LIMENG':'limited_eng',
    'E_MUNIT':'10_units_plus',
    'E_MOBILE':'mobile_homes',
    'E_CROWD':'crowded_units',
    'E_NOVEH':'no_vehicle',
    'E_GROUPQ':'persons_group_quarters',
    'EP_POV':'pct_pov',
    'EP_UNEMP':'pct_unemp',
    'EP_PCI':'pct_cost_burdened_low_income',
})

evictions_geo = evictions_geocoded[['latitude','longitude']]
evictions_geo = evictions_geo.dropna()

evictions_geo
m = fm.Map([34,-118])

evictions_geocoded.info()

HeatMap(evictions_geo).add_to(m)
m.save('map.html')

svi_la.head()