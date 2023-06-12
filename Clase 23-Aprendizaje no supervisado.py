#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:28:10 2023

@author: clinux01
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1
# =============================================================================
archivo =  '/home/clinux01/Descargas/datos_clase_clustering.csv' 
data = pd.read_csv(archivo)

#%%
# =============================================================================
# 2 Construir la variable ppm que calcule el precio por metro cuadrado cubierto.
# =============================================================================
data['ppm'] = data['price']/data['surface_covered']

#%%
# =============================================================================
# 3 Visualizar los valores de ppm po r ubicacion geografica, graficar las viviendas en lo sejes lat-lon, con colores segun ppm
# =============================================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(data = data , x = 'lon' , y = 'lat' , hue= 'ppm', palette='viridis').set(title='Valors de PPM por ubicacion geografica')
plt.show()

#%%
# =============================================================================
# 4 Agrupar los datos utilizando el metodo kmeans. considerando distintos conjuntos de variables.
# =============================================================================

X = data.loc[:,['lon','lat','ppm']]

kmeans = KMeans(n_clusters=3,random_state=0,n_init='auto').fit(X)
y_kmeans = kmeans.predict(X)
kmeans.labels_

data['predict'] = y_kmeans
plt.figure(figsize=(8, 6))
sns.scatterplot(data = data , x = 'lon' , y = 'lat' , hue= 'predict',palette='viridis').set(title='Valors de PPM por ubicacion geografica')
plt.show()

# Separa bien con 3 clusters el ppm por ubicacion geografica
# =============================================================================
# Probar con distintos valores de k, cantidad de clusters.
# =============================================================================
#%%
data.dropna(inplace=True)
X = data.loc[:,['lon','lat','rooms','ppm']]

kmeans = KMeans(n_clusters=5,random_state=0,n_init='auto').fit(X)
y_kmeans = kmeans.predict(X)
data['predict'] = y_kmeans
plt.figure(figsize=(8, 6))
sns.scatterplot(data = data , x = 'lon' , y = 'lat' , hue= 'predict',palette= 'viridis').set(title='Valors de PPM por ubicacion geografica')
plt.show()

# Se separa mal porque hay rooms 

#%%
# =============================================================================
# Ejercicio 6
# Visualizar los clusters obtenidos. Para denotar el agrupamiento asignado en un scatterplot, construir 
# una variable nueva donde se indique el agrupamiento asignado, y utilizarla para dar color a los puntos
# en el scatterplot.
# =============================================================================

plt.figure(figsize=(8, 6))
sns.scatterplot(data = data , x = 'surface_covered' , y = 'price' , hue= 'predict',palette= 'viridis').set(title='Valors de PPM por ubicacion geografica')
plt.show()


#%%

# =============================================================================
# 5
# =============================================================================

kmeans = KMeans(n_clusters=4,random_state=0,n_init='auto').fit(X)
y_kmeans = kmeans.predict(X)
data['predict'] = y_kmeans
plt.figure(figsize=(8, 6))
sns.scatterplot(data = data , x = 'lon' , y = 'lat' , hue= 'predict',palette= 'viridis').set(title='Valors de PPM por ubicacion geografica')
plt.show()

#%%
kmeans = KMeans(n_clusters=5,random_state=0,n_init='auto').fit(X)
y_kmeans = kmeans.predict(X)
data['predict'] = y_kmeans
plt.figure(figsize=(8, 6))
sns.scatterplot(data = data , x = 'lon' , y = 'lat' , hue= 'predict',palette= 'viridis').set(title='Valors de PPM por ubicacion geografica')
plt.show()

# =============================================================================
# Ejercicio 7
# Visualizar los agrupamientos pero segun la ubicacion geografica, graficar las viviendas en los ejes lon-lat, con colores segun el cluster asignado.
# =============================================================================

