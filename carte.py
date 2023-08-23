import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Application de Carte avec GeoPandas et Streamlit")

# Chargement du fichier Shapefile avec GeoPandas
@st.cache  # Cette ligne permet de mettre en cache les données pour une meilleure performance
def load_shapefile(shapefile_path):
    return gpd.read_file(shapefile_path)

shapefile_path = "SEN_adm3.shp"
gdf = load_shapefile(shapefile_path)

# Affichage de la carte à l'aide de Matplotlib
st.subheader("Carte affichée avec Matplotlib")
fig, ax = plt.subplots()
gdf.plot(ax=ax)
st.pyplot(fig)

# Affichage des données attributaires en tant que tableau
st.subheader("Données attributaires")
st.write(gdf)

# Affichage des statistiques des données attributaires
st.subheader("Statistiques des données attributaires")
st.write(gdf.describe())

# Affichage des informations géométriques
st.subheader("Informations géométriques")
st.write(gdf.geometry.describe())

# Affichage des premières lignes des données attributaires
st.subheader("Premières lignes des données attributaires")
st.write(gdf.head())

