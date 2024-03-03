# %% [markdown]
# # Evictions in LA 2023

# %% [markdown]
# ## Import

# %%
import pandas as pd
import geopandas as gpd 
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from libpysal.weights import KNN
import esda
from esda.moran import Moran
import splot
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster,plot_moran_simulation
import libpysal 
import seaborn as sns
import contextily
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import glm
from patsy import dmatrices
from pygris import tracts
import bambi as bmb
import arviz as az


# %%

evictions = pd.read_csv('data/2023_eviction_filings_final.csv')
evictions_geocoded = pd.read_csv('data/evictions_geocoded.csv')
svi_la = pd.read_csv('data/SVI_la.csv',dtype={'FIPS':str})
rent_df = pd.read_csv('data/ACSST5Y2020.S2502-Data.csv',dtype={'FIPS':str})


# %%

gdf_ct = tracts(state='California',county='Los Angeles')

# %% [markdown]
# ## Cleaning

# %%

# Convert evictions 'Address' to string
evictions['Address'] = evictions['Address'].astype(str)

# Create new column in evictions_geocode with matching name 'Address' and make sure it's a string
evictions_geocoded['Address'] = evictions_geocoded['input_string'].astype(str)
evictions_geocoded = gpd.GeoDataFrame(evictions_geocoded,geometry=gpd.points_from_xy(evictions_geocoded['longitude'],evictions_geocoded['latitude']),crs='4326')
evics = evictions_geocoded.merge(evictions,left_index=True,right_index=True)
evics = evics.to_crs(epsg='3857')




# %%
rent_df = rent_df[['FIPS','S2502_C05_001E']]
rent_df = rent_df.rename(columns={'S2502_C05_001E':'renter_occupied'})

# %%
gdf_ct.head()

# %%
svi_la = svi_la[svi_la['COUNTY'] == 'Los Angeles']

# %%
gdf_ct = gdf_ct.to_crs(epsg='3857')

# trim the data to the bare minimum columns
gdf_ct = gdf_ct[['GEOID','geometry']]

# rename the columns
gdf_ct.columns = ['FIPS','geometry']
# last rows
gdf_ct.plot()



# %%
svi_la['FIPS'].isin(gdf_ct['FIPS'])

# %% [markdown]
# ### Process and merge dataframes

# %%
# Merge tract geo data with social vulnerability index data
gdf_ct = gdf_ct.merge(svi_la,on='FIPS')

# Select and rename columns
gdf_ct = gdf_ct.loc[:, ['FIPS','E_TOTPOP', 'E_HU', 'E_HH', 'EP_POV150', 'EP_UNEMP','EP_NOHSDP', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 'EP_SNGPNT', 'EP_MINRTY', 'EP_LIMENG', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ', 'geometry']]
gdf_ct = gdf_ct.rename(
    columns={'E_TOTPOP':'total_pop','E_HU':'housing_units','E_HH':'households','EP_POV150':'pov_below150','EP_UNEMP':'unemployed','EP_PCI':'cost_burdened_low_income','EP_NOHSDP':'no_high_school','EP_AGE65':'persons_over65','EP_AGE17':'persons_under17','EP_DISABL':'disabled_pop','EP_SNGPNT':'single_parent','EP_MINRTY':'minority_pop','EP_LIMENG':'limited_eng','EP_MUNIT':'10_units_plus','EP_MOBILE':'mobile_homes','EP_CROWD':'crowded_units','EP_NOVEH':'no_vehicle','EP_GROUPQ':'persons_group_quarters'
})

# Join cleaned up dataset with evictions using spatial join. Also create a new column of 'evictions_count' to count per tract
join = gdf_ct.sjoin(evics)
evics_ct = join.FIPS.value_counts().rename_axis('FIPS').reset_index(name='evictions_count')
gdf_ct=gdf_ct.merge(evics_ct,on='FIPS')

# Merge our dataframe with dataset containing total number of rental units per census tract
gdf_ct = gdf_ct.merge(rent_df,on='FIPS')

# Remove tracts without rental units and scale by 1000 
gdf_ct = gdf_ct[gdf_ct['renter_occupied']>0]
gdf_ct['count_norm'] = gdf_ct['evictions_count'] / gdf_ct['renter_occupied'] * 100
gdf_ct

# %% [markdown]
# ### Preliminary visualization of outcome variable

# %%
gdf_ct.columns.to_list()

# %% [markdown]
# Non parametric testing because of difference in mean and median

# %% [markdown]
# ## Data Exploration

# %%
gdf_ct['count_norm'].describe()

# %% [markdown]
# ### Scale data using a min max scaler

# %%
census_vars = [
 'total_pop',
 'housing_units',
 'households',
 'pov_below150',
 'unemployed',
 'no_high_school',
 'persons_over65',
 'persons_under17',
 'disabled_pop',
 'single_parent',
 'minority_pop',
 'limited_eng',
 '10_units_plus',
 'mobile_homes',
 'crowded_units',
 'no_vehicle',
 'persons_group_quarters',
 'evictions_count',
 'renter_occupied',
 'count_norm'
]

# %%
scaler = MinMaxScaler()
gdf_ct[census_vars] = scaler.fit_transform(gdf_ct[census_vars])


# %%
table = gdf_ct.describe()
table

# %%

corr = gdf_ct[census_vars].corr()
plt.figure(figsize=(10,9))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.show()

# %% [markdown]
# ### Calculate variance inflation factor for each variable considering high collinearity based on data exploration

# %%
C = add_constant(gdf_ct[census_vars])

vif = pd.Series([variance_inflation_factor(C.values,i) for i in range(C.shape[1])],index=C.columns)
vif

# %%
gdf_ct.hist(bins=15,figsize=(15,10))
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(10,2,figsize=(15,10))
ax = ax.flatten()

for i, var in enumerate(gdf_ct[census_vars]):
    sns.boxplot(y=var,data=gdf_ct,ax=ax[i])
   
plt.show()

# %%
sns.pairplot(gdf_ct[census_vars],kind='reg')
plt.show()

# %%
config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [
        {
          "dataId": [
            "evictions"
          ],
          "id": "sreoedtdg",
          "name": [
            "Notice Date"
          ],
          "type": "timeRange",
          "value": [
            1598918400000,
            1601078400000
          ],
          "enlarged": True,
          "plotType": "histogram",
          "yAxis": None
        }
      ],
      "layers": [
        {
          "id": "y8t676q",
          "type": "grid",
          "config": {
            "dataId": "evictions",
            "label": "Point",
            "color": [
              34,
              63,
              154
            ],
            "columns": {
              "lat": "latitude",
              "lng": "longitude"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "worldUnitSize": 0.5,
              "colorRange": {
                "name": "ColorBrewer RdYlGn-6",
                "type": "diverging",
                "category": "ColorBrewer",
                "colors": [
                  "#1a9850",
                  "#91cf60",
                  "#d9ef8b",
                  "#fee08b",
                  "#fc8d59",
                  "#d73027"
                ],
                "reversed": True
              },
              "coverage": 1,
              "sizeRange": [
                0,
                500
              ],
              "percentile": [
                0,
                100
              ],
              "elevationPercentile": [
                0,
                100
              ],
              "elevationScale": 20.9,
              "colorAggregation": "count",
              "sizeAggregation": "count",
              "enable3d": True
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": None,
            "colorScale": "quantize",
            "sizeField": None,
            "sizeScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "evictions": [
              {
                "name": "Notice Date",
                "format": None
              },
            ]
          },
          "compareMode": False,
          "compareType": "absolute",
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 24,
      "dragRotate": True,
      "latitude": 33.837184166330836,
      "longitude": -118.46478962372794,
      "pitch": 50,
      "zoom": 9,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": True,
        "water": True,
        "land": True,
        "3d building": False
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

# %%
from keplergl import KeplerGl
map = KeplerGl(height=600, width=800, data={'evictions':evics},config=config)
map

# %% [markdown]
# ### Summary of exploration
# 
# So it looks as though the data is not normally distributed across multiple variables and several variables have high degrees of multicollinearity. The low numbers of mobile homes and group quarters also seem to skew the data. I'm not entirely sure how to address these but will return to this. 

# %% [markdown]
# ## COVID Rent Protections and timeline of evictions

# %%
# Convert notice date to datetime format
evics['notice_date_dt'] = pd.to_datetime(evics['Notice Date'], format='%m/%d/%y')

# Make sure there aren't any from outside 2023
evics = evics[evics['notice_date_dt'].dt.year >= 2023 ]

# Create a dataframe to hold eviction counts per day
address_count_grpd = evics.value_counts(['notice_date_dt']).reset_index(name='count')
address_count_grpd


# %% [markdown]
# ### Timeline of evictions for 2023

# %%
fig_date_year = px.bar(
    address_count_grpd,
    x = 'notice_date_dt',
    y='count',
    text='count',
    labels={
        'notice_date_dt':'Notice Date',
        'count':'Count'
    },
    title='Eviction filings in 2023',
    
)
fig_date_year.add_annotation(x='2023-04-01', y=0,
            text="End of COVID Rent Protections",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=-20,
            ay=-200,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ffffff",
            opacity=0.8)
fig_date_year.show()

# %% [markdown]
# ## Spatial Analysis

# %% [markdown]
# ### Calculate spatial weights

# %%
# Generate W from the GeoDataFrame
w = KNN.from_dataframe(gdf_ct,geom_col='geometry',k=8)
# Row-standardization
w.transform = "R"
gdf_ct["w_count"] = libpysal.weights.lag_spatial(w, gdf_ct['count_norm'])

# %%
lisa = esda.moran.Moran_Local(gdf_ct['count_norm'], w)
# Draw KDE line
ax = sns.kdeplot(lisa.Is)
# Add one small bar (rug) for each observation
# along horizontal axis
sns.rugplot(lisa.Is, ax=ax);

# %% [markdown]
# ### Create moran scatterplot

# %%
# Set y to list of normalized eviction count
y = gdf_ct['count_norm']

# Initialize moran plot and then graph
moran = Moran(y,w)
fig, ax = moran_scatterplot(moran, aspect_equal=True)
plt.show()

# %% [markdown]
# ### Plot Reference Distribution

# %%
plot_moran_simulation(moran,aspect_equal=False)

# %% [markdown]
# ### Visualize weighted count vs original, normalized count

# %%
f, axs = plt.subplots(1,2,figsize=(20, 16))
ax1, ax2 = axs

gdf_ct.plot(
    column="w_count",
    cmap="viridis",
    scheme="quantiles",
    k=5,
    edgecolor="white",
    linewidth=0.0,
    alpha=0.75,
    legend=True,
    ax=ax1,
)

gdf_ct.plot(
    column="count_norm",
    cmap="viridis",
    scheme="quantiles",
    k=5,
    edgecolor="white",
    linewidth=0.0,
    alpha=0.75,
    legend=True,
    ax=ax2,
)

ax1.axis('off')
ax2.axis('off')
ax1.set_title('Evictions Per 1000 Renter Occupied Units - Spatial Lag')
ax2.set_title('Evictions Per 1000 Renter Occupied Units')
plt.show()

# %% [markdown]
# ### Calculate local moran values, plot values, and visualize clusters

# %%
# calculate local moran values
lisa = esda.moran.Moran_Local(y, w)

# %%
# Plot
fig,ax = plt.subplots(figsize=(20,20))

moran_scatterplot(lisa, ax=ax, p=0.05)
ax.set_xlabel("Evictions")
ax.set_ylabel('Spatial Lag of Evictions')

# add some labels
plt.text(1.95, 0.5, "HH", fontsize=25)
plt.text(1.95, -1, "HL", fontsize=25)
plt.text(-2, 1, "LH", fontsize=25)
plt.text(-1, -1, "LL", fontsize=25)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(14,12))
lisa_cluster(lisa, gdf_ct, p=0.05, ax=ax)
plt.show()

# %% [markdown]
# And create a map comparing different p-values

# %%
# create the 1x2 subplots
fig, ax = plt.subplots(1, 2, figsize=(20, 12))

# regular count map on the left
lisa_cluster(lisa, gdf_ct, p=0.05, ax=ax[0])

ax[0].axis("off")
ax[0].set_title("P-value: 0.05")

# spatial lag map on the right
lisa_cluster(lisa, gdf_ct, p=0.01, ax=ax[1])
ax[1].axis("off")
ax[1].set_title("P-value: 0.01")

plt.show()

# %%
# Create a binary variable to subset 'high' eviction rate tracts and 'low' eviction rate tracts
evic_mean = gdf_ct['count_norm'].mean()
gdf_ct['evic_risk'] = gdf_ct['count_norm']>evic_mean

# Visualize the new variable
f, ax = plt.subplots(1,figsize=(10,10))

gdf_ct.plot(
    ax=ax,
    column='evic_risk',
    categorical=True,
    legend=True,
    colormap='Set3',
)
ax.set_axis_off()
plt.show()

# %%
# Transform
w.transform = "O"
jc = esda.join_counts.Join_Counts(gdf_ct['evic_risk'], w)
jc_table = pd.DataFrame(data=[
    [jc.bb,jc.bw,jc.ww,(jc.bb+jc.bw+jc.ww)],
    [jc.mean_bb,jc.mean_bw,'na','na'],
    [jc.p_sim_bb,jc.p_sim_bw,'na','na']],
    columns= ['Low-Low','Low-High','High-High','Sum'],index=['Actual', 'Predicted', 'p-values'])
jc_table

# %%
gdf_ct = gdf_ct.dropna()

# %%
gdf_ct.shape

# %%
fml = 'count_norm ~ pov_below150 + unemployed + no_high_school + limited_eng + minority_pop + disabled_pop + single_parent + limited_eng + crowded_units'

y, X = dmatrices(fml,gdf_ct,return_type='dataframe')

X = add_constant(X)
model = sm.GLM(y,X,family=sm.families.Gaussian())
results = model.fit()
print(results.summary())

# %%

bayes_model = bmb.Model(formula=fml,data=gdf_ct,dropna=True)

bayes_model.prior_predictive
# model_fitted = bayes_model.fit()
# az.plot_trace(model_fitted, compact=True)
# az.summary(model_fitted)



