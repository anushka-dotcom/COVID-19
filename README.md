# COVID-19
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive/')
df = pd.read_csv('drive/My Drive/Datasets/covid/covid_19_clean_complete.csv', parse_dates= ['Date'], index_col = 'Date')
df.head()
df.tail()
df.shape
df.isnull().sum()
df.dtypes
df.describe()
wrc = df.groupby('WHO Region')['Confirmed'].sum().reset_index()
print(wrc)
labels = df['WHO Region'].unique()
plt.figure(figsize = (8, 8))
colors = ['deepskyblue', 'steelblue', 'cadetblue', 'aqua', 'teal', 'mediumaquamarine']

plt.pie(wrc['Confirmed'], autopct = '%1.1f%%', labels = labels, explode = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03), colors = colors)

plt.title('Confirmed cases in WHO Regions', size = 15, fontweight = 'bold')
plt.show()
wrr = df.groupby('WHO Region')['Recovered'].sum().reset_index()
print(wrr)
labels = df['WHO Region'].unique()
plt.figure(figsize = (8, 8))
colors = ['violet', 'purple', 'mediumvioletred', 'crimson', 'magenta', 'darkviolet']

plt.pie(wrr['Recovered'], autopct = '%1.1f%%', labels = labels, explode = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03), colors = colors)

plt.title('Recovered cases in WHO Regions', size = 15, fontweight = 'bold')
plt.show()
wrx = df.groupby('WHO Region')['Deaths'].sum().reset_index()
print(wrx)
labels = df['WHO Region'].unique()
plt.figure(figsize = (8, 8))
colors = ['red', 'darkred', 'firebrick', 'crimson', 'orangered', 'tomato']

plt.pie(wrx['Deaths'], autopct = '%1.1f%%', labels = labels, explode = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03), colors = colors)

plt.title('Death cases in WHO Regions', size = 15, fontweight = 'bold')
plt.show()
em = df[df['WHO Region'] == 'Eastern Mediterranean']
u = df[df['WHO Region'] == 'Europe']
af = df[df['WHO Region'] == 'Africa']
am = df[df['WHO Region'] == 'Americas']
wp = df[df['WHO Region'] == 'Western Pacific']
sea = df[df['WHO Region'] == 'South-East Asia']
rem = em.resample('M').mean()
ru = u.resample('M').mean()
ref = af.resample('M').mean()
ram = am.resample('M').mean()
rwp = wp.resample('M').mean()
rsea = sea.resample('M').mean()
plt.figure(figsize = (15, 7))

plt.plot(rem['Confirmed'], label = 'Eastern Mediterranean', linewidth = 2)
plt.plot(ru['Confirmed'], label = 'Europe', linewidth = 2)
plt.plot(ref['Confirmed'], label = 'Africa', linewidth = 2)
plt.plot(ram['Confirmed'], label = 'Americas', linewidth = 2)
plt.plot(rwp['Confirmed'], label = 'Western Pacific', linewidth = 2)
plt.plot(rsea['Confirmed'], label = 'South-East Asia', linewidth = 2)

plt.title('Confirmed Cases with respect to Months\n', size = 16, fontweight = 'bold')
plt.xlabel('\nMonth', size = 11, fontweight = 'bold')
plt.ylabel('Confirmed Cases\n', size = 11, fontweight = 'bold')

plt.legend()
plt.show()

plt.figure(figsize = (15, 7))

plt.plot(rem['Recovered'], label = 'Eastern Mediterranean', linewidth = 2)
plt.plot(ru['Recovered'], label = 'Europe', linewidth = 2)
plt.plot(ref['Recovered'], label = 'Africa', linewidth = 2)
plt.plot(ram['Recovered'], label = 'Americas', linewidth = 2)
plt.plot(rwp['Recovered'], label = 'Western Pacific', linewidth = 2)
plt.plot(rsea['Recovered'], label = 'South-East Asia', linewidth = 2)

plt.title('Recovered Cases with respect to Months\n', size = 16, fontweight = 'bold')
plt.xlabel('\nMonth', size = 11, fontweight = 'bold')
plt.ylabel('Recovered Cases\n', size = 11, fontweight = 'bold')

plt.legend()
plt.show()
plt.figure(figsize = (15, 7))

plt.plot(rem['Deaths'], label = 'Eastern Mediterranean', linewidth = 2)
plt.plot(ru['Deaths'], label = 'Europe', linewidth = 2)
plt.plot(ref['Deaths'], label = 'Africa', linewidth = 2)
plt.plot(ram['Deaths'], label = 'Americas', linewidth = 2)
plt.plot(rwp['Deaths'], label = 'Western Pacific', linewidth = 2)
plt.plot(rsea['Deaths'], label = 'South-East Asia', linewidth = 2)

plt.title('Death Cases with respect to Months\n', size = 16, fontweight = 'bold')
plt.xlabel('\nMonth', size = 11, fontweight = 'bold')
plt.ylabel('Death Cases\n', size = 11, fontweight = 'bold')

plt.legend()
plt.show()
sacr = sea.groupby('Country/Region')['Confirmed'].sum().reset_index()
print(sacr)
plt.figure(figsize = (12, 8))

sacc = sns.barplot('Country/Region', 'Confirmed', data = sacr)

for p in sacc.patches:
    sacc.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

plt.show()
sacrr = sea.groupby('Country/Region')['Recovered'].sum().reset_index()
print(sacrr)
plt.figure(figsize = (12, 8))

saccr = sns.barplot('Country/Region', 'Recovered', data = sacrr)

for p in saccr.patches:
    saccr.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

plt.show()
sacrrx = sea.groupby('Country/Region')['Deaths'].sum().reset_index()
print(sacrrx)
plt.figure(figsize = (12, 8))

saccrx = sns.barplot('Country/Region', 'Deaths', data = sacrrx)

for p in saccrx.patches:
    saccrx.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

plt.show()
bd = df[df['Country/Region'] == 'Bangladesh']
plt.figure(figsize = (15, 6))

plt.plot(bd['Confirmed'], linewidth = 3)

plt.xlabel('\nDay', size = 12, fontweight = 'bold')
plt.ylabel('Confirmed Cases\n', size = 12, fontweight = 'bold')

plt.show()
bd_m = bd.resample('M').mean()

plt.figure(figsize = (15, 6))

plt.plot(bd_m['Confirmed'], linewidth = 3)

plt.xlabel('\nMonths', size = 12, fontweight = 'bold')
plt.ylabel('Confirmed Cases\n', size = 12, fontweight = 'bold')

plt.show()
plt.figure(figsize = (15, 6))

plt.plot(bd['Recovered'], linewidth = 3, color = 'green')

plt.xlabel('\nDay', size = 12, fontweight = 'bold')
plt.ylabel('Recovered Cases\n', size = 12, fontweight = 'bold')

plt.show()

plt.plot(bd_m['Recovered'], linewidth = 3, color = 'green')

plt.xlabel('\nMonths', size = 12, fontweight = 'bold')
plt.ylabel('Recovered Cases\n', size = 12, fontweight = 'bold')

plt.show()
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['Long'], df['Lat']))
gdf.head(3)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize = (20, 10))
gdf.plot(cmap = 'Reds', ax = ax)
world.geometry.boundary.plot(color = None, edgecolor = 'k', linewidth = 2, ax = ax)
wp = world[world['continent'] == 'Asia']

fig, ax = plt.subplots(figsize = (20, 10))
gdf[gdf['Country/Region'] == 'China'].plot(cmap = 'Reds', ax = ax)
wp.geometry.boundary.plot(color = None, edgecolor = 'k', linewidth = 2, ax = ax)
