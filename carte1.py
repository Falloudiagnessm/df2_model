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
# Liste des régions disponibles dans le fichier Shapefile
regions = gdf['NAME_1'].unique()
selected_region = st.selectbox("Sélectionnez une région", regions)

# Filtrer la région sélectionnée
diourbel_region = gdf[gdf['NAME_1'] == selected_region]

# Filtrer les régions de Diourbel et Thiès
#diourbel_region = gdf[gdf['NAME_1'] == 'Diourbel']
thies_region = gdf[gdf['NAME_1'] == 'Thiès']

# Couleur pour la région de Diourbel (rouge)
diourbel_color = 'red'
# Couleur pour la région de Thiès (bleu)
thies_color = 'blue'

# Création de la carte avec GeoPandas
st.subheader("Carte de Diourbel et Thiès colorées")
fig, ax = plt.subplots()
gdf.plot(ax=ax, color='gray')  # Rendre les autres régions en gris par défaut
diourbel_region.plot(ax=ax, color=diourbel_color)  # Colorer la région de Diourbel en rouge
thies_region.plot(ax=ax, color=thies_color)  # Colorer la région de Thiès en bleu
st.pyplot(fig)


# Affichage des données attributaires, statistiques, etc. (comme dans votre code)
#st.subheader("Données attributaires de Diourbel")
#st.write(Diourbel_region)

#st.subheader("Statistiques des données attributaires de Dakar")
#st.write(Diourbel_region.describe())

#st.subheader("Informations géométriques de Dakar")
#st.write(Diourbel_region.geometry.describe())

#st.subheader("Premières lignes des données attributaires de Dakar")
#st.write(Diourbel_region.head())

