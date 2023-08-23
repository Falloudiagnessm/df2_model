import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Application de Carte avec GeoPandas et Streamlit")

# Chargement du fichier Shapefile avec GeoPandas
@st.cache
def load_shapefile(shapefile_path):
    return gpd.read_file(shapefile_path)

shapefile_path = "SEN_adm3.shp"
gdf = load_shapefile(shapefile_path)

# Filtrer la région de Dakar
dakar_region = gdf[gdf['NAME_1'] == 'Thiès']
st.write(dakar_region)
# Couleur pour la région de Dakar (rouge)
dakar_color = 'red'

# Création de la carte avec GeoPandas
st.subheader("Carte de Dakar colorée en rouge")
fig, ax = plt.subplots()
gdf.plot(ax=ax, color='gray')  # Rendre les autres régions en gris par défaut
dakar_region.plot(ax=ax, color=dakar_color)  # Colorer la région de Dakar en rouge
st.pyplot(fig)

# Affichage des données attributaires, statistiques, etc. (comme dans votre code)
st.subheader("Données attributaires de Dakar")
st.write(dakar_region)

st.subheader("Statistiques des données attributaires de Dakar")
st.write(dakar_region.describe())

st.subheader("Informations géométriques de Dakar")
st.write(dakar_region.geometry.describe())

st.subheader("Premières lignes des données attributaires de Dakar")
st.write(dakar_region["NAME_3"])
#dakar_region["NAME_3]



