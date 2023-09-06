import streamlit as st
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize , ListedColormap,BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

# sudo sysctl fs.inotify.max_user_watches=1000000

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
    t = np.arange(0, 200, 1)
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
    
    # Création d'un DataFrame à partir des données de la simulation (simulation fictive)
 
    results_df = pd.DataFrame({'Temps': t, 'Infectés I1 Zone1': u[:, 2], 'Infectés I2 Zone1': u[:, 3], 'Infectés I1 Zone2':u[:, 10],'Infectés I2 Zone2': u[:, 11]})
    st.write(" affichage du dataframe")
    st.write(results_df)
    infectes_zone_1 = results_df['Infectés I1 Zone1']+results_df['Infectés I2 Zone1']
    infectes_zone_2 = results_df['Infectés I1 Zone2']+results_df['Infectés I2 Zone2']
 
 
 
    val1=[(infectes_zone_1[j]-min(infectes_zone_1))/(max(infectes_zone_1)-min(infectes_zone_1)) for j in range(len(infectes_zone_1))]
 
    val2=[(infectes_zone_2[j]-min(infectes_zone_2))/(max(infectes_zone_2)-min(infectes_zone_2)) for j in range(len(infectes_zone_2))]
 
    
    
    # Créer les couleurs basées sur le nombre d'infectés
    color1_data = [f'#{int(v * 255):02X}0000' for v in val1]
    color2_data = [f'#{int(v * 255):02X}0000' for v in val2]
 
 
    st.subheader("Affichage de la carte d'infection")
 
    # Chargement du fichier Shapefile avec GeoPandas
    @st.cache_data
    def load_shapefile(shapefile_path):
        return gpd.read_file(shapefile_path)

    shapefile_path = "SEN_adm3.shp"
    gdf = load_shapefile(shapefile_path)

    # Liste des arrondissements uniques dans le jeu de données de Thies
    thies_arrondissements = gdf[gdf['NAME_1'] == 'Thiès']['NAME_3'].unique()
 
    fig, ax = plt.subplots()
 
    gdf[gdf['NAME_1'] == 'Thiès'].plot(ax=ax, color='gray')  # Rendre les autres arrondissements en gris par défaut

 
     
    # Sélection des arrondissements à afficher
    selected_arrondissements = st.multiselect("Sélectionnez les arrondissements", thies_arrondissements,default=["Fissel","Sessene"])
    # Créer une colormap personnalisée
    cmap_bar = ListedColormap(["#FF0000","#CC0000","#BB0000","#990000","#660000" ])
 
 
    cmap = ListedColormap(color1_data+color2_data)
 
 
    # Créer une Normalize pour les valeurs d'infections
    norm = Normalize(vmin=min(infectes_zone_1), vmax=max(infectes_zone_2))

    # Créer un objet ScalarMappable avec la colormap et la Normalize
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #breakpoints = [0,min(infectes_zone_1)+165,min(infectes_zone_1)+330,500]
    max_value =max(infectes_zone_2)
    breakpoints = [0,min(infectes_zone_1)+100,min(infectes_zone_1)+200,min(infectes_zone_1)+300,min(infectes_zone_1)+400]
    norm_a_personnaliser = BoundaryNorm(breakpoints, cmap_bar.N)
    sm_bar = plt.cm.ScalarMappable(cmap=cmap_bar, norm = norm_a_personnaliser)
 
 
    sm.set_array([]) # Pas nécessaire, mais évite un avertissement
  
    # Parcourir les arrondissements sélectionnés
    for idx, selected_arrondissement in enumerate(selected_arrondissements):
         arrondissement_data = gdf[(gdf['NAME_1'] == 'Thiès') & (gdf['NAME_3'] == selected_arrondissement)]
         
         # Sélectionner les couleurs en fonction de la zone
         if idx == 0:
            arrondissement_data.plot(ax=ax, color="#FF0000", legend=True)
         else :
            arrondissement_data.plot(ax=ax, color="#660000", legend=True)
       
    if selected_arrondissements==[]:
        st.write("veuiller selectionner deux zone à etudier")
  
    # Ajouter une barre de couleur (colorbar) pour indiquer les valeurs d'infections
    cbar = plt.colorbar(sm_bar, ax=ax, format='%d')
    cbar.set_label('Infection Cases')

    # Add a "+" sign after the last value in the legend
    # Add a "+" sign to all values in the colorbar
    colorbar_ticks = cbar.get_ticks()
    last_tick = colorbar_ticks[-1]
    new_labels = [f'+{int(tick)}' if tick == last_tick else f'{int(tick)}' for tick in colorbar_ticks]
    cbar.ax.set_yticklabels(new_labels)
    
    st.title("Carte des arrondissements de Thies")
    
 

   
    for arrondissement, color in [(selected_arrondissements[0], color1_data), (selected_arrondissements[1], color2_data)]:
        arrondissement_data = gdf[(gdf['NAME_1'] == 'Thiès') & (gdf['NAME_3'] == arrondissement)]
        x = arrondissement_data.geometry.centroid.x.values[0]
        y = arrondissement_data.geometry.centroid.y.values[0]
        ax.text(x, y, arrondissement, fontsize=9, color="green")

    # Ajouter une légende
    ax.legend(loc='upper left')    
   
    # Afficher la carte à l'aide de Matplotlib dans Streamlit
    st.pyplot(fig)

 
    return t , u 
    
# Interface utilisateur Streamlit
st.title('Modèle de transmission de la peste entre rongeurs et puces')

# Widgets d'entrée
st.sidebar.subheader(" Population initiale")
 
#Initial1 = [500, 0, 50, 70, 10, 20, 60, 80]
#Initial2 = [ 0, 500, 0, 0, 0, 0, 0, 0]
# Champ de saisie pour les valeurs initiales
Initial1 = st.sidebar.text_input('Population initiale zone 1 (S1,S2,I1,I2,A1,A2,L1,L2)', value="500,0,50,70,10,20,60,80")
Initial1 = [float(val.strip()) for val in Initial1.split(',')]

Initial2 = st.sidebar.text_input('Population initiale zone 2 (S1,S2,I1,I2,A1,A2,L1,L2)', value=" 0,500,0,0,0,0,0,0")
Initial2 = [float(val.strip()) for val in Initial2.split(',')] 
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

st.sidebar.write("le taux de mortalité des puces libres")
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

# Bouton pour exécuter le modèle
#if st.button('Exécuter le modèle'):
t, u = model_diffusion(Initial1,Initial2,epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2, delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1, beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F)
 
# Création d'un DataFrame à partir des données de la simulation
#results_df = pd.DataFrame({'Temps': t, 'Infectés Zone 1': u[:, 2], 'Infectés Zone 2': u[:, 6]})
# Afficher le DataFrame
#st.write('Résultats du modèle :')
#st.dataframe(results_df)


 














 
