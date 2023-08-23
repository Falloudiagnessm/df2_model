import streamlit as st
import folium
from folium import plugins
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import geopandas as gpd

st.set_option('deprecation.showPyplotGlobalUse', False)


def hcv4(y, t, epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2,beta_3, beta_4, d_1, d_2, gamma_1,gamma_2, delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2,K2_R, alpha2, beta2_1, beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2,delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F):
    dy = np.zeros(16)

    # Nombre total de rongeurs zone 1
    N = y[0] + y[1] + y[2] + y[3]

    # Zone 1
    dy[0] = ((1 - epsilon) * r_1 * y[0] + epsilon * r_2 * y[1]) * (1 - (N / K_R)) - y[0] * (1 - np.exp(-alpha * N)) * (beta_1 * (y[6] / N) + beta_2 * (y[7] / N)) - d_1 * y[0] + gamma_1 * y[2] - a_12 * y[0] + a_21 * y[8]
    dy[1] = ((1 - epsilon) * r_2 * y[1] + epsilon * r_1 * y[0]) * (1 - (N / K_R)) - y[1] * (1 - np.exp(-alpha * N)) * (beta_3 * (y[6] / N) + beta_4 * (y[7] / N)) - d_2 * y[1] + gamma_2 * y[3] - a_12 * y[1] + a_21 * y[9]
    dy[2] = y[0] * (1 - np.exp(-alpha * N)) * (beta_1 * (y[6] / N) + beta_2 * (y[7] / N)) - (d_1 + delta_1) * y[2] - gamma_1 * y[2] - a_12 * y[2] + a_21 * y[10]
    dy[3] = y[1] * (1 - np.exp(-alpha * N)) * (beta_3 * (y[6] / N) + beta_4 * (y[7] / N)) - (d_2 + delta_2) * y[3] - gamma_1 * y[3] - a_12 * y[3] + a_21 * y[11]
    dy[4] = ra_1 * y[4] * (1 - (y[4] / K_A)) + (y[6] * (1 - np.exp(-alpha * N))) / N
    dy[5] = ra_2 * y[5] * (1 - (y[5] / K_A)) + (y[7] * (1 - np.exp(-alpha * N))) / N
    dy[6] = (d_1 + delta_1) * y[2] * y[4] - y[6] * (1 - np.exp(-alpha * N)) - d_F * y[6] - b_12 * y[6] + b_21 * y[14]
    dy[7] = (d_2 + delta_2) * y[3] * y[5] - y[7] * (1 - np.exp(-alpha * N)) - d_F * y[7] - b_12 * y[7] + b_21 * y[15]

    # Zone 2
    N1 = y[8] + y[9] + y[10] + y[11]
    dy[8] = ((1 - epsilon) * r2_1 * y[8] + epsilon * r2_2 * y[9]) * (1 - (N1 / K2_R)) - y[8] * (1 - np.exp(-alpha2 * N1)) * (beta2_1 * (y[14] / N1) + beta2_2 * (y[15] / N1)) - d2_1 * y[8] + gamma2_1 * y[10] + a_12 * y[0] - a_21 * y[8]
    dy[9] = ((1 - epsilon) * r2_2 * y[9] + epsilon * r2_1 * y[8]) * (1 - (N1 / K2_R)) - y[9] * (1 - np.exp(-alpha2 * N1)) * (beta2_3 * (y[14] / N1) + beta2_4 * (y[15] / N1)) - d2_2 * y[9] + gamma2_2 * y[11] + a_12 * y[1] - a_21 * y[9]
    dy[10] = y[8] * (1 - np.exp(-alpha2 * N1)) * (beta2_1 * (y[14] / N1) + beta2_2 * (y[15] / N1)) - (d2_1 + delta2_1) * y[10] - gamma2_1 * y[10] + a_12 * y[2] - a_21 * y[10]
    dy[11] = y[9] * (1 - np.exp(-alpha2 * N1)) * (beta2_3 * (y[14] / N1) + beta2_4 * (y[15] / N1)) - (d2_2 + delta2_2) * y[11] - gamma2_1 * y[11] + a_12 * y[3] - a_21 * y[11]
    dy[12] = ra2_1 * y[12] * (1 - (y[12] / K2_A)) + (y[14] * (1 - np.exp(-alpha2 * N))) / N1
    dy[13] = ra2_2 * y[13] * (1 - (y[13] / K2_A)) + (y[15] * (1 - np.exp(-alpha2 * N))) / N1
    dy[14] = (d2_1 + delta2_1) * y[10] * y[12] - y[14] * (1 - np.exp(-alpha2 * N1)) - d2_F * y[14] + b_12 * y[6] - b_21 * y[14]
    dy[15] = (d2_2 + delta2_2) * y[11] * y[13] - y[15] * (1 - np.exp(-alpha2 * N1)) - d2_F * y[15] + b_12 * y[7] - b_21 * y[15]

    return dy


# Définition de la fonction Mon_model pour générer les graphiques et la carte
def model_diffusion(Initial1,Initial2,epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2, delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1, beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F):
    t = np.arange(0, 300, 0.5)
    # Initial1 = [votre liste d'initialisation]
    
    Initial=Initial1+Initial2
    #Initial = [500, 0, 50, 70, 10, 20, 60, 80, 0, 500, 0, 0, 0, 0, 0, 0]

    # Utilisez odeint pour résoudre les équations différentielles
    u = odeint(hcv4, Initial, t, args=(epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2, delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1, beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F))


    
    st.subheader("affichage des graphiques")
    #affichage des graphiques
    # Dataframes for visualization
    
    df1 = pd.DataFrame({'t': t, 'S_1': u[:, 0], 'S_2': u[:, 1], 'I_1': u[:, 2], 'I_2': u[:, 3]})
    df2 = pd.DataFrame({'t': t, 'S_1': u[:, 8], 'S_2': u[:, 9], 'I_1': u[:, 10], 'I_2': u[:, 11]})
    df3 = pd.DataFrame({'t': t, 'A_1': u[:, 4], 'A_2': u[:, 5], 'L_1': u[:, 6], 'L_2': u[:, 7]})
    df4 = pd.DataFrame({'t': t, 'A_1': u[:, 12], 'A_2': u[:, 13], 'L_1': u[:, 14], 'L_2': u[:, 15]})
    
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot p1
    axs[0, 0].plot(df1['t'], df1['S_1'], color='blue',label='S_1')
    axs[0, 0].plot(df1['t'], df1['S_2'], color='green',label='S_2')
    axs[0, 0].plot(df1['t'], df1['I_1'], color='red',label='I_1')
    axs[0, 0].plot(df1['t'], df1['I_2'], color='black',label='I_1')
    axs[0, 0].set_title('Evolution des rongeurs dans la zone 1')
    axs[0, 0].set_xlabel('time (jours)')
    axs[0, 0].set_ylabel('Proportion des rongeurs')
    axs[0, 0].legend()

    # Plot p2
    axs[0, 1].plot(df2['t'], df2['S_1'],'--', color='blue',label='S_1')
    axs[0, 1].plot(df2['t'], df2['S_2'], '--',color='green',label='S_2')
    axs[0, 1].plot(df2['t'], df2['I_1'],'--', color='red',label='I_1')
    axs[0, 1].plot(df2['t'], df2['I_2'],'--', color='black',label='I_2')
    axs[0, 1].set_title('Evolution des rongeurs dans la zone 2')
    axs[0, 1].set_xlabel('time (jours)')
    axs[0, 1].set_ylabel('Proportion des rongeurs')
    axs[0, 1].legend()

    # Plot p3
    axs[1, 0].plot(df3['t'], df3['A_1'], color='blue',label='A_1')
    axs[1, 0].plot(df3['t'], df3['A_2'], color='green',label='A_2')
    axs[1, 0].plot(df3['t'], df3['L_1'], color='red',label='L_1')
    axs[1, 0].plot(df3['t'], df3['L_2'], color='black',label='L_2')
    axs[1, 0].set_title('Evolution des puces dans la zone 1')
    axs[1, 0].set_xlabel('time (jours)')
    axs[1, 0].set_ylabel('Proportion des puces')
    axs[1, 0].legend()

    # Plot p4
    axs[1, 1].plot(df4['t'], df4['A_1'],'--', color='blue',label='A_1')
    axs[1, 1].plot(df4['t'], df4['A_2'],'--', color='green',label='A_2')
    axs[1, 1].plot(df4['t'], df4['L_1'],'--', color='red',label='L_1')
    axs[1, 1].plot(df4['t'], df4['L_2'],'--', color='black',label='L_2')
    axs[1, 1].set_title('Evolution des puces dans la zone 2')
    axs[1, 1].set_xlabel('time (jours)')
    axs[1, 1].set_ylabel('Proportion des puces')
    axs[1, 1].legend()
    # Adjust layout
    plt.tight_layout()

    # Show the plots
    st.pyplot()



    st.subheader("Affichage de la carte d'infection")
    # Calculer les zones les plus touchées 
 
    max_infection_zone1 = np.sum(u[:, 2])+np.sum(u[:, 3])+np.sum(u[:, 6])+np.sum(u[:, 7])
 
    max_infection_zone2 = np.sum(u[:, 10])+np.sum(u[:, 11])+np.sum(u[:,14])+np.sum(u[:,15])
    #st.write("Maximum d'infections dans la zone 1 :", max_infection_zone1)
    #st.write("Maximum d'infections dans la zone 1 :", max_infection_zone2)


    most_affected_zone = 1 if max_infection_zone2 > max_infection_zone1 else 2
 

    # Chargement du fichier Shapefile avec GeoPandas
    @st.cache_data
    def load_shapefile(shapefile_path):
        return gpd.read_file(shapefile_path)

    shapefile_path = "SEN_adm3.shx"
    gdf = load_shapefile(shapefile_path)

    # Liste des arrondissements uniques dans le jeu de données de Thies
    thies_arrondissements = gdf[gdf['NAME_1'] == 'Thiès']['NAME_3'].unique()

    # Sélection des deux arrondissements à afficher
    selected_arrondissements = st.selectbox("Sélectionnez le premier arrondissement", thies_arrondissements), st.selectbox("Sélectionnez le   deuxième arrondissement", thies_arrondissements)

    # Création de la carte avec GeoPandas
    st.title("Carte des arrondissements de Thies")
    fig, ax = plt.subplots()
    gdf[gdf['NAME_1'] == 'Thiès'].plot(ax=ax, color='gray')  # Rendre les autres arrondissements en gris par défaut

    # Filtrer les données pour les arrondissements sélectionnés
    arrondissement1_data = gdf[(gdf['NAME_1'] == 'Thiès') & (gdf['NAME_3'] == selected_arrondissements[0])]
    arrondissement2_data = gdf[(gdf['NAME_1'] == 'Thiès') & (gdf['NAME_3'] == selected_arrondissements[1])]

    # Couleurs en fonction de la zone la plus touchée
    zone_color1 = 'blue' if most_affected_zone == 1 else 'red'
    zone_color2 = 'red' if most_affected_zone == 1 else 'blue'

    # Plot des arrondissements sélectionnés avec les couleurs appropriées
    arrondissement1_data.plot(ax=ax, color=zone_color1 )
    arrondissement2_data.plot(ax=ax, color=zone_color2 )
    
    
    # Ajouter les noms des arrondissements avec les couleurs correspondantes
    for arrondissement, color in [(selected_arrondissements[0], zone_color1), (selected_arrondissements[1], zone_color2)]:
        arrondissement_data = gdf[(gdf['NAME_1'] == 'Thiès') & (gdf['NAME_3'] == arrondissement)]
        x = arrondissement_data.geometry.centroid.x.values[0]
        y = arrondissement_data.geometry.centroid.y.values[0]
        ax.text(x, y, arrondissement, fontsize=9, color="green")

    # Ajouter une légende
    ax.legend(loc='upper left')
 
 
     
     
    st.pyplot(fig)

 
 

 
    st.write('NB: La zone colorée en rouge est plus touchée par l infection des puces sur les rongeurs')

    # Création de la carte avec Folium
    #m = folium.Map(location=[48.8566, 2.3522], zoom_start=6)
    coords_SN = [14.6928, -17.4467]
    #coords_thies = [(14.7924, -16.9550)]
    m = folium.Map(location=coords_SN, zoom_start=8)

    # Coordonnées des zones
 
    # Coordonnées du Sénégal et du Mali
    #coords_senegal = [(14.692778, -17.446667), (14.692778, -12.9), (16.691944, -17.446667), (16.691944, -12.9)]
    #coords_mouritanie = [(17.0, -12.9), (17.0, -7.0), (24.0, -12.9), (24.0, -7.0)]
    #coords_mali = [(11.4, -12.5), (11.4, -4.0), (24.0, -12.5), (24.0, -4.0)]
    
    #coords_thies = [(14.7924, -16.9550)]
    #coords_dakar = [(14.6928, -17.4467)]
    
    


    # Appliquer la couleur rouge à la zone la plus infectée
    #color_mali = 'red' if most_affected_zone == 1 else 'blue'
    #color_senegal = 'blue' if most_affected_zone == 1 else 'red'
    
    # Ajouter des marqueurs pour le Sénégal et le Mali
    #folium.Marker(location=[14.6928, -17.4467], popup='Sénégal', icon=folium.Icon(color='blue')).add_to(m)
    #folium.Marker(location=[17.5707, -3.9962], popup='Mali', icon=folium.Icon(color='red')).add_to(m)
    
    #polygones
    #folium.Polygon(coords_senegal, color=color_senegal, fill=True, fill_color=color_senegal, fill_opacity=0.4).add_to(m)
    #folium.Polygon(coords_mali, color=color_mali, fill=True, fill_color=color_mali, fill_opacity=0.4).add_to(m)
    
    # Ajouter des popups aux marqueurs
    #folium.Marker(location=[14.6928, -17.4467], popup='Sénégal', icon=folium.Icon(color='blue')).add_to(m)
    #folium.Marker(location=[17.5707, -3.9962], popup='Mali', icon=folium.Icon(color='red')).add_to(m)
    
    # Ajouter des cercles autour des zones d'infection maximale
    #folium.Circle(location=[14.6928, -17.4467], radius=max_infection_zone1 * 1000, color='blue', fill=True, fill_color='blue', fill_opacity=0.3).add_to(m)
    #folium.Circle(location=[17.5707, -3.9962], radius=max_infection_zone2 * 1000, color='red', fill=True, fill_color='red', fill_opacity=0.3).add_to(m)

 
    # Ajouter des marqueurs pour les zones de Thiès et Dakar
    #for coord in coords_thies:
    #folium.CircleMarker(location=coords_zone1[0], radius=10, color=color_thies, fill=True, fill_opacity=0.7).add_to(m)

    #for coord in coords_dakar:
    #folium.CircleMarker(location=coords_zone2[1], radius=10, color=color_dakar, fill=True, fill_opacity=0.7).add_to(m)


    # Afficher la carte dans Streamlit
    #folium_static(m)
    Thiès_Est = [(14.7812711,-16.9116994)]
    Thiès_Ouest = [(14.7823473,-16.9418797)]
    Thiès_Nord =[(14.8050219,-16.9238989)]
    Thiès_Sud = [(14.791461, -16.925605)]
    
    coords_zone1=[(14.7812711,-16.9116994),(14.7823473,-16.9418797),(14.8050219,-16.9238989),(14.791461, -16.925605)]
    coords_zone2=[(14.7812711,-16.9116994),(14.7823473,-16.9418797),(14.8050219,-16.9238989),(14.791461, -16.925605)]

    
    # Couleurs en fonction de la zone la plus touchée
    color_zone1 = 'red' if most_affected_zone == 1 else 'blue'
    color_zone2 = 'blue' if most_affected_zone == 1 else 'red'
 
 
    # Initialiser la carte Folium
    #m = folium.Map(location=coords_thies[0], zoom_start=13)

    # Ajouter les marqueurs de zone 1
    #for coord in coords_zone1:
    #    folium.CircleMarker(location=coord, radius=10, color='blue', fill=True, fill_opacity=0.7).add_to(m)

    #  Ajouter les marqueurs de zone 2
    #for coord in coords_zone2:
     #   folium.CircleMarker(location=coord, radius=10, color='green', fill=True, fill_opacity=0.7).add_to(m)

    # Afficher la carte Folium dans Streamlit
    st.title("Carte des zones")
    st.write("Sélectionnez une zone dans la liste déroulante :")

    # Liste déroulante pour sélectionner la zone
    
    selected_zone1 = st.selectbox("Sélectionnez la zone 1", ["Thiès Est", "Thiès Ouest", "Thiès Nord", "Thiès Sud"])
    
    selected_zone2 = st.selectbox("Sélectionnez la zone 2", ["Thiès Est", "Thiès Ouest", "Thiès Nord", "Thiès Sud"])
    
    if selected_zone1 == "Thiès Est" and selected_zone2 == "Thiès Ouest":
        folium.CircleMarker(location=coords_zone1[0], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[1], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Est" and selected_zone2 == "Thiès Nord":
        folium.CircleMarker(location=coords_zone1[0], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[2], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Est" and selected_zone2 == "Thiès Sud":
        folium.CircleMarker(location=coords_zone1[0], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[3], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
        
        
    elif selected_zone1 == "Thiès Ouest" and selected_zone2 == "Thiès Est":
        folium.CircleMarker(location=coords_zone1[1], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[0], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Ouest" and selected_zone2 == "Thiès Nord":
        folium.CircleMarker(location=coords_zone1[1], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[2], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Ouest" and selected_zone2 == "Thiès Sud":
        folium.CircleMarker(location=coords_zone1[1], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[3], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
        
        
    elif selected_zone1 == "Thiès Nord" and selected_zone2 == "Thiès Est":
        folium.CircleMarker(location=coords_zone1[2], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[0], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Nord" and selected_zone2 == "Thiès Ouest":
        folium.CircleMarker(location=coords_zone1[2], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[1], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Nord" and selected_zone2 == "Thiès Sud":
        folium.CircleMarker(location=coords_zone1[2], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[3], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
        
    elif selected_zone1 == "Thiès Sud" and selected_zone2 == "Thiès Est":
        folium.CircleMarker(location=coords_zone1[3], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[0], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Sud" and selected_zone2 == "Thiès Ouest":
        folium.CircleMarker(location=coords_zone1[3], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[1], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium
    elif selected_zone1 == "Thiès Sud" and selected_zone2 == "Thiès Nord":
        folium.CircleMarker(location=coords_zone1[3], radius=10, color=color_zone1, fill=True, fill_opacity=0.7).add_to(m)
        folium.CircleMarker(location=coords_zone2[2], radius=10, color=color_zone2, fill=True, fill_opacity=0.7).add_to(m)
        folium_static(m)  # Afficher la carte Folium        
    else:
        st.write('veuillez choisir deux zones differentes')    
        
          
  

def main():
    st.title("Modèle mathématique de transmission de la peste: diffusion entre deux zones")
    
    st.sidebar.subheader("valeurs intitiales")
    #Initial1 = [500, 0, 50, 70, 10, 20, 60, 80]
    #Initial2 = [ 0, 500, 0, 0, 0, 0, 0, 0]
    # Champ de saisie pour les valeurs initiales
    Initial1 = st.sidebar.text_input('Population initiale zone 1 (S1,S2,I1,I2,A1,A2,L1,L2)', value="500,0,50,70,10,20,60,80")
    Initial1 = [float(val.strip()) for val in Initial1.split(',')]
    
    Initial2 = st.sidebar.text_input('Population initiale zone 2 (S1,S2,I1,I2,A1,A2,L1,L2)', value=" 0,500,0,0,0,0,0,0")
    Initial2 = [float(val.strip()) for val in Initial2.split(',')]
    # Créer les sliders interactifs pour les valeurs initiales
    #initial1 = [st.number_input(f'Initial1 {i+1}', min_value=0, max_value=1000, value=Initial1[i]) for i in range(len(Initial1))]
    #initial2 = [st.number_input(f'Initial2 {i+1}', min_value=0, max_value=1000, value=Initial2[i]) for i in range(len(Initial2))]

    # Slider widgets to adjust the parameters
    st.sidebar.subheader("les paramettre pour la zone 1")
    epsilon = st.sidebar.slider("epsilon", 0.0, 1.0, 0.3)
    epsilon= st.sidebar.number_input("epsilon", min_value=0.0, max_value=1.0, value=epsilon)
    st.sidebar.write("les taux de reproduction des rongeurs")
    r_1 = st.sidebar.slider("r_1", 0.0, 1.0, 0.9)
    r_1 = st.sidebar.number_input("r_1", min_value=0.0, max_value=1.0, value=r_1)

    r_2 = st.sidebar.slider("r_2", 0.0, 1.0, 0.79)
    r_2 = st.sidebar.number_input("r_2", min_value=0.0, max_value=1.0, value=r_2)
    st.sidebar.write("la capacité limite des rongeurs")
    K_R = st.sidebar.slider("K_R", 0.0, 2000.0, 1000.0)
    K_R= st.sidebar.number_input("K_R", min_value=0.0, max_value=2000.0, value=K_R)
    st.sidebar.write("paramettre qui mesure l'efficacité de recherche des rongeurs")
    alpha = st.sidebar.slider("alpha", 0.0, 1.0, 0.7)
    alpha = st.sidebar.number_input("alpha", min_value=0.0, max_value=1.0, value=alpha)
    st.sidebar.write("les paramettres de transmision")
    beta_1 = st.sidebar.slider("beta_1", 0.0, 1.0, 0.1)
    beta_1 = st.sidebar.number_input("beta_1", min_value=0.0, max_value=1.0, value=beta_1)
    beta_2 = st.sidebar.slider("beta_2", 0.0, 1.0, 0.1)
    beta_2 = st.sidebar.number_input("beta_2", min_value=0.0, max_value=1.0, value=beta_2)
    beta_3 = st.sidebar.slider("beta_3", 0.0, 1.0, 0.1)
    beta_3 = st.sidebar.number_input("beta_3", min_value=0.0, max_value=1.0, value=beta_3)
    beta_4 = st.sidebar.slider("beta_4", 0.0, 1.0, 0.3)
    beta_4 = st.sidebar.number_input("beta_4", min_value=0.0, max_value=1.0, value=beta_4)
    st.sidebar.write("les taux de mortalités naturels des rongeurs")
    d_1 = st.sidebar.slider("d_1", 0.0, 1.0, 0.05)
    d_1 = st.sidebar.number_input("d_1", min_value=0.0, max_value=1.0, value=d_1)
    d_2 = st.sidebar.slider("d_2", 0.0, 1.0, 0.025)
    d_2 = st.sidebar.number_input("d_2", min_value=0.0, max_value=1.0, value=d_2)
    
    st.sidebar.write("les taux de guerrisons des rongeurs")
    gamma_1 = st.sidebar.slider("gamma_1", 0.0, 1.0, 0.8)
    gamma_1 = st.sidebar.number_input("gamma_1", min_value=0.0, max_value=1.0, value=gamma_1)

    gamma_2 = st.sidebar.slider("gamma_2", 0.0, 1.0, 0.8)
    gamma_2 = st.sidebar.number_input("gamma_2", min_value=0.0, max_value=1.0, value=gamma_2)
    st.sidebar.write("les taux de mortalité du à la maladie")
    delta_1 = st.sidebar.slider("delta_1", 0.0, 1.0, 0.1)
    delta_1 = st.sidebar.number_input("delta_1", min_value=0.0, max_value=1.0, value=delta_1)

    delta_2 = st.sidebar.slider("delta_2", 0.0, 1.0, 0.1)
    delta_2 = st.sidebar.number_input("delta_2", min_value=0.0, max_value=1.0, value=delta_2)
    
    st.sidebar.write("les taux de reproduction des puces attachées aux rongeurs ")
    ra_1 = st.sidebar.slider("ra_1", 0.0, 1.0, 0.3)
    ra_1 = st.sidebar.number_input("ra_1", min_value=0.0, max_value=1.0, value=ra_1)

    ra_2 = st.sidebar.slider("ra_2", 0.0, 1.0, 0.4)
    ra_2 = st.sidebar.number_input("ra_2", min_value=0.0, max_value=1.0, value=ra_2)
    st.sidebar.write("la capacité limite des puces attachées aux rongeurs")
    K_A = st.sidebar.slider("K_A", 0, 100, 50)
    K_A = st.sidebar.number_input("K_A", min_value=0, max_value=100, value=K_A)
    st.sidebar.write("taux de mortalité des puces libres")
    d_F = st.sidebar.slider("d_F", 0.0, 1.0, 0.9)
    d_F = st.sidebar.number_input("d_F", min_value=0.0, max_value=1.0, value=d_F)
 
    st.sidebar.subheader("les paramettre pour la zone 2")
    
    st.sidebar.write("les taux de reproduction des rongeurs")
    r2_1 = st.sidebar.slider("r2_1", 0.0, 1.0, 0.9)
    r2_1 = st.sidebar.number_input("r2_1", min_value=0.0, max_value=1.0, value=r2_1)
    r2_2 = st.sidebar.slider("r2_2", 0.0, 1.0, 0.79)
    r2_2 = st.sidebar.number_input("r2_2", min_value=0.0, max_value=1.0, value=r2_2)
    st.sidebar.write("la capacité limite des rongeurs")
    K2_R = st.sidebar.slider("K2_R", 0, 2000, 1000)
    K2_R = st.sidebar.number_input("K2_R", min_value=0, max_value=2000, value=K2_R)

    alpha2 = st.sidebar.slider("alpha2", 0.0, 1.0, 0.7)
    alpha2 = st.sidebar.number_input("alpha2", min_value=0.0, max_value=1.0, value=alpha2)
    
    st.sidebar.write("les paramettres de transmision")
    beta2_1 = st.sidebar.slider("beta2_1", 0.0, 1.0, 0.2)
    beta2_1 = st.sidebar.number_input("beta2_1", min_value=0.0, max_value=1.0, value=beta2_1)

    beta2_2 = st.sidebar.slider("beta2_2", 0.0, 1.0, 0.2)
    beta2_2 = st.sidebar.number_input("beta2_2", min_value=0.0, max_value=1.0, value=beta2_2)

    beta2_3 = st.sidebar.slider("beta2_3", 0.0, 1.0, 0.2)
    beta2_3 = st.sidebar.number_input("beta2_3", min_value=0.0, max_value=1.0, value=beta2_3)
    
 

    beta2_4 = st.sidebar.slider("beta2_4", 0.0, 1.0, 0.4)
    beta2_4 = st.sidebar.number_input("beta2_4", min_value=0.0, max_value=1.0, value=beta2_4)
    
    st.sidebar.write("les taux de mortalités naturels des rongeurs")

    d2_1 = st.sidebar.slider("d2_1", 0.0, 1.0, 0.04)
    d2_1 = st.sidebar.number_input("d2_1", min_value=0.0, max_value=1.0, value=d2_1)

    d2_2 = st.sidebar.slider("d2_2", 0.0, 1.0, 0.035)
    d2_2 = st.sidebar.number_input("d2_1", min_value=0.0, max_value=1.0, value=d2_2)
    st.sidebar.write("les taux de guerrisons des rongeurs")
    gamma2_1 = st.sidebar.slider("gamma2_1", 0.0, 1.0, 0.7)
    gamma2_1 = st.sidebar.number_input("gamma2_1", min_value=0.0, max_value=1.0, value=gamma2_1)

    gamma2_2 = st.sidebar.slider("gamma2_2", 0.0, 1.0, 0.7)
    gamma2_2 = st.sidebar.number_input("gamma2_2", min_value=0.0, max_value=1.0, value=gamma2_2)

    delta2_1 = st.sidebar.slider("delta2_1", 0.0, 1.0, 0.2)
    delta2_1 = st.sidebar.number_input("delta2_1", min_value=0.0, max_value=1.0, value=delta2_1)
    
    st.sidebar.write("les taux de mortalité du à la maladie")
    delta2_2 = st.sidebar.slider("delta2_2", 0.0, 1.0, 0.2)
    delta2_2 = st.sidebar.number_input("delta2_2", min_value=0.0, max_value=1.0, value=delta2_2)
    
    st.sidebar.write("les taux de reproduction des puces attachées aux rongeurs ")
    ra2_1 = st.sidebar.slider("ra2_1", 0.0, 1.0, 0.4)
    ra2_1 = st.sidebar.number_input("ra2_1", min_value=0.0, max_value=1.0, value=ra2_1)

    ra2_2 = st.sidebar.slider("ra2_2", 0.0, 1.0, 0.5)
    ra2_2 = st.sidebar.number_input("ra2_2", min_value=0.0, max_value=1.0, value=ra2_2)
    
    st.sidebar.write("la capacité limite des puces attaché aux rongeurs")
    K2_A = st.sidebar.slider("K2_A", 0, 100, 50)
    K2_A = st.sidebar.number_input("K2_A", min_value=0, max_value=100, value=K2_A)
    
    st.sidebar.write("le de mortalité des puces libres")
    d2_F = st.sidebar.slider("d2_F", 0.0, 1.0, 0.8)
    d2_F = st.sidebar.number_input("d2_F", min_value=0.0, max_value=1.0, value=d2_F)

    
    st.sidebar.subheader("Paramettre de migrations")
    
    #a_12 = st.slider("A_12", 0.0, 0.01, 0.0008)
    #a_12 = st.number_input("a_12", min_value=0.0, max_value=1.0, value=a_12)

    #a_21 = st.slider("A_21", 0.0, 0.01, 0.0008)
    #a_21 = st.number_input("a_21", min_value=0.0, max_value=1.0, value=a_21)

    #b_12 = st.slider("B_12", 0.0, 0.001, 0.0002)
    #b_12 = st.number_input("b_12", min_value=0.0, max_value=1.0, value=b_12)

    #b_21 = st.slider("B_21", 0.0, 0.001, 0.0004)
    #b_21 = st.number_input("b_21", min_value=0.0, max_value=1.0, value=b_21)

    a_12 = st.sidebar.slider("a_12", 0.0, 1.0, 0.08)
    a_12 = st.sidebar.number_input("a_12", min_value=0.0, max_value=1.0, value=a_12)
    
    a_21 = st.sidebar.slider("a_21", 0.0, 0.01, 0.0008)
    a_21 = st.sidebar.number_input("a_21", min_value=0.0, max_value=1.0, value=a_21)

    b_12 = st.sidebar.slider("b_12", 0.0, 1.00, 0.02)
    b_12 = st.sidebar.number_input("b_12", min_value=0.0, max_value=1.0, value=b_12)

    b_21 = st.sidebar.slider("b_21", 0.0, 0.001, 0.0004)
    b_21 = st.sidebar.number_input("b_21", min_value=0.0, max_value=1.0, value=b_21)

    # Bouton d'exécution
    #run_button = st.button("Exécuter le modèle")

    #if run_button:
        # Appel à la fonction model_diffusion avec les paramètres ajustés
        #model_diffusion(Initial1,Initial2,epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2, delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1, beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F)
    model_diffusion(Initial1,Initial2,epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2, delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1, beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F)

if __name__ == "__main__":
    main()

