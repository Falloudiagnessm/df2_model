import streamlit as st
########
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
#######
import streamlit.components.v1 as html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize , ListedColormap,BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fsolve
######
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots 
import plotly.express as px

# sudo sysctl fs.inotify.max_user_watches=1000000
# Augmenter la limite à 300 Mo (vous pouvez ajuster cette valeur)
#
#st.set_option('server.maxMessageSize', 300)

#py.init_notebook_mode(connected=True)




################################################################################################################################
########################################### hide streamlit components ##########################################################
################################################################################################################################

hide_menu_style = """ 
        <style> 
        #MainMenu {visibility : hidden; }
        footer {visibility : hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html = True)


################################################################################################################################
################################################ barre latérale ################################################################
################################################################################################################################


with st.sidebar:
    choose = option_menu("App Model", ["Accueil", "modèle", "rapport", "Code source"],
                         icons=['house', 'clipboard-data', 'clipboard-data', 'clipboard-data'],
                         menu_icon="app-indicator",default_index=0, 
                         styles={                  
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#3379FF"},
    }
    )
    
    
    
    
################################################################################################################################
################################################ page d'accueil ################################################################
################################################################################################################################

if choose == "Accueil" :
        
        
        
        # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Accueil</p>', unsafe_allow_html=True)        
        st.markdown("""
        Dans le cadre de notre etude, nous avons exploré la puissance de Streamlit,
        un framework Python, pour créer rapidement une application interactive pour notre etude sur la modélisation
         de la transmission de la peste entre rongeurs et puces pour les trois trois zones ciblées. Cette démarche
        a permis de simplifier le processus de développement, permettant ainsi aux développeurs de
        concevoir des applications web complexes en seulement quelques lignes de code.
        Notre objectif principal était de modéliser la transmission de la peste dans trois zones géo-
        graphiques distinctes, en utilisant un modèle mathématique approprié. Les trois principales
        étapes de ce projet sont resumées comme suit :""")
        st.subheader('Traitement du modèle')
        st.markdown(""" Nous avons commencé par définir un modèle mathéma-
        tique représentant la dynamique des populations et la propagation des rongeurs infectés
        dans les différentes zones. Ce modèle était basé sur des équations différentielles ordi
        naires (EDO) et des données réelles ou estimées pour les paramètres. """)
        #import image
        image_path = "zone1.png"   
        st.image(image_path, caption='Dynamique du modèle dans la zone 1', use_column_width=True)
        image_path = "zone2.png"   
        st.image(image_path, caption='Dynamique du modèle dans la zone 2', use_column_width=True)
        image_path = "zone3.png"   
        st.image(image_path, caption='Dynamique du modèle dans la zone 3', use_column_width=True)

        st.subheader('Utilisation des API de streamlit')
        st.markdown(""" Ensuite, nous avons exploité les API de Streamlit pour créer une interface utilisateur 
        interactive. Les utilisateurs peuvent désormais ajuster les paramètres du modèle, lancer des simulations et 
        visualiser les résultats de manière conviviale. """)
        
        st.subheader('Déploiement de l application Web  ')
        st.markdown(""" Nous avons déployé notre application sur la plateforme Streamlit Sharing, la rendant ainsi
         accessible en ligne. Vous pouvez désormais explorer l'application. Cette etude démontre comment la modélisation et la simulation de phénomènes complexes peuvent être rendues accessibles à un large public grâce à des outils comme Streamlit. Nous espérons que cette application facilitera la compréhension de la transmission de la peste et encouragera de nouvelles recherches dans ce domaine.""")

st.set_option('deprecation.showPyplotGlobalUse', False)


################################################################################################################################
################################################ modèle ##################################################################
################################################################################################################################    

if choose == "modèle":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modèle de transmission de la peste entre hotes et vecteurs</p>', unsafe_allow_html=True)
     
    #with st.expander(label="Veuillez cliquer pour déplier/replier"):
    
    def system_edo(y, t, epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2, delta_1,
                   delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1, beta2_2, beta2_3,
                   beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
                   beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32):
        dy = np.zeros(24)

        # Nombre total de rongeurs pour chaque zones
        N = y[0] + y[1] + y[2] + y[3]
        N1 = y[8] + y[9] + y[10] + y[11]
        N3 = y[16] + y[17] + y[18] + y[19]


        # Zone 1
        dy[0] = ((1 - epsilon) * r_1 * y[0] + epsilon * r_2 * y[1]) * (1 - (N / K_R)) - y[0] * (1 - np.exp(-alpha * N)) * (beta_1 * (y[6] / N) + beta_2 * (y[7] / N)) - d_1 * y[0] + gamma_1 * y[2] - a_12 * y[0] + a_21 *  y[8] - a_13 * y[0]   + a_31 *  y[16]
        dy[1] = ((1 - epsilon) * r_2 * y[1] + epsilon * r_1 * y[0]) * (1 - (N / K_R)) - y[1] * (1 - np.exp(-alpha * N)) * (beta_3 * (y[6] / N) + beta_4 * (y[7] / N)) - d_2 * y[1] + gamma_2 * y[3] - a_12 * y[1] + a_21 *  y[9] - a_13 * y[1]   + a_31 *  y[17]
        dy[2] = y[0] * (1 - np.exp(-alpha * N)) * (beta_1 * (y[6] / N) + beta_2 * (y[7] / N)) - (d_1 + delta_1) * y[2] - gamma_1 * y[2] - a_12 * y[2] + a_21 * y[10] - a_13 * y[2] + a_31 * y[18]
        dy[3] = y[1] * (1 - np.exp(-alpha * N)) * (beta_3 * (y[6] / N) + beta_4 * (y[7] / N)) - (d_2 + delta_2) * y[3] - gamma_1 * y[3] - a_12 * y[3] + a_21 * y[11] - a_13 * y[3] + a_31 * y[19]
        dy[4] = ra_1 * y[4] * (1 - (y[4] / K_A)) + (y[6] * (1 - np.exp(-alpha * N))) / N
        dy[5] = ra_2 * y[5] * (1 - (y[5] / K_A)) + (y[7] * (1 - np.exp(-alpha * N))) / N
        dy[6] = (d_1 + delta_1) * y[2] * y[4] - y[6] * (1 - np.exp(-alpha * N)) - d_F * y[6] - b_12 * y[6] + b_21 * y[14] - b_13 * y[6] + b_31 * y[22]
        dy[7] = (d_2 + delta_2) * y[3] * y[5] - y[7] * (1 - np.exp(-alpha * N)) - d_F * y[7] - b_12 * y[7] + b_21 * y[15] - b_13 * y[7] + b_31 * y[23]

        # Zone 2

        dy[8] = ((1 - epsilon) * r2_1 * y[8] + epsilon * r2_2 * y[9]) * (1 - (N1 / K2_R)) - y[8] * (1 - np.exp(-alpha2 * N1)) * (beta2_1 * (y[14] / N1) + beta2_2 * (y[15] / N1)) - d2_1 * y[8] + gamma2_1 * y[10] - a_21 * y[8] + a_12 * y[0] - a_23 * y[8] + a_32 * y[16]
        dy[9] = ((1 - epsilon) * r2_2 * y[9] + epsilon * r2_1 * y[8]) * (1 - (N1 / K2_R)) - y[9] * (1 - np.exp(-alpha2 * N1)) * (beta2_3 * (y[14] / N1) + beta2_4 * (y[15] / N1)) - d2_2 * y[9] + gamma2_2 * y[11] - a_21 * y[9] + a_12 * y[1] - a_23 * y[9] + a_32 * y[17]
        dy[10] = y[8] * (1 - np.exp(-alpha2 * N1)) * (beta2_1 * (y[14] / N1) + beta2_2 * (y[15] / N1)) - (d2_1 + delta2_1) * y[10] - gamma2_1 * y[10] - a_21 * y[10] + a_12 * y[2] - a_23 * y[10] + a_32 * y[18]
        dy[11] = y[9] * (1 - np.exp(-alpha2 * N1)) * (beta2_3 * (y[14] / N1) + beta2_4 * (y[15] / N1)) - (d2_2 + delta2_2) * y[11] - gamma2_1 * y[11] - a_21 * y[11] + a_12 * y[3] - a_23 * y[11] + a_32 * y[19]
        dy[12] = ra2_1 * y[12] * (1 - (y[12] / K2_A)) + (y[14] * (1 - np.exp(-alpha2 * N))) / N1
        dy[13] = ra2_2 * y[13] * (1 - (y[13] / K2_A)) + (y[15] * (1 - np.exp(-alpha2 * N))) / N1
        dy[14] = (d2_1 + delta2_1) * y[10] * y[12] - y[14] * (1 - np.exp(-alpha2 * N1)) - d2_F * y[14] - b_21 * y[14] + b_12 * y[6] - b_23 * y[14] + b_32 * y[22]
        dy[15] = (d2_2 + delta2_2) * y[11] * y[13] - y[15] * (1 - np.exp(-alpha2 * N1)) - d2_F * y[15] - b_21 * y[15] + b_12 * y[7] - b_23 * y[15] + b_32 * y[23]

        #  zone 3


        dy[16] = ((1 - epsilon) * r3_1 * y[16] + epsilon * r3_2 * y[17]) * (1 - (N3 / K3_R)) - y[16] * (1 - np.exp(-alpha3 * N3)) * (beta3_1 * (y[22] / N3) + beta3_2 * (y[23] / N3)) - d3_1 * y[16] + gamma3_1 * y[18] - a_31 * y[16] + a_23 * y[8] - a_32 * y[16]   + a_13 * y[0]
        dy[17] = ((1 - epsilon) * r3_2 * y[17] + epsilon * r3_1 * y[16]) * (1 - (N3 / K3_R)) - y[17] * (1 - np.exp(-alpha3 * N3)) * (beta3_3 * (y[22] / N3) + beta3_4 * (y[23] / N3)) - d3_2 * y[17] + gamma3_2 * y[19] - a_31 * y[17] + a_23 * y[9] - a_32 * y[17]   + a_13 * y[1]
        dy[18] = y[16] * (1 - np.exp(-alpha3 * N3)) * (beta3_1 * (y[22] / N3) + beta3_2 * (y[23] / N3)) - (d3_1 + delta3_1) * y[18] - gamma3_1 * y[18] - a_31 * y[18] + a_23 * y[10] - a_32 * y[18] + a_13 * y[2]
        dy[19] = y[17] * (1 - np.exp(-alpha3 * N3)) * (beta3_3 * (y[22] / N3) + beta3_4 * (y[23] / N3)) - (d3_2 + delta3_2) * y[19] - gamma3_1 * y[19] - a_31 * y[19] + a_23 * y[11] - a_32 * y[19] + a_13 * y[3]
        dy[20] = ra3_1 * y[20] * (1 - (y[20] / K3_A)) + (y[22] * (1 - np.exp(-alpha3 * N3))) / N3
        dy[21] = ra3_2 * y[21] * (1 - (y[21] / K3_A)) + (y[23] * (1 - np.exp(-alpha3 * N3))) / N3
        dy[22] = (d3_1 + delta3_1) * y[18] * y[20] - y[22] * (1 - np.exp(-alpha3 * N3)) - d3_F * y[22] - b_31 * y[22] + b_23 * y[14] - b_32 * y[22] + b_13 * y[6]
        dy[23] = (d3_2 + delta3_2) * y[19] * y[21] - y[23] * (1 - np.exp(-alpha3 * N3)) - d3_F * y[23] - b_31 * y[23] + b_23 * y[15] - b_32 * y[23] + b_13 * y[7]

        return dy

    # Définition de la fonction Mon_model pour générer les graphiques et la carte


    def model_prague(temps_final, Initial1,Initial2,Initial3, epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2,
                     delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1,
                     beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
                   beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32):
        # Appel du system_edo pour la résolution
        t = np.arange(0, temps_final, 1)
        Initial=Initial1+Initial2+Initial3

        u = odeint(system_edo, Initial, t, args=(epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2,
                                                  gamma_1, gamma_2, delta_1, delta_2,  ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1,
                     beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
                   beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32))

       ## return t, u    
############################################################################################################################################
###########################################################################################################################################
########################################## Chargement des fichiers de SIG ############################################################
######################################################################################################################################       
       
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
        selected_arrondissements = st.multiselect("Sélectionnez les arrondissements", thies_arrondissements,default=["Fissel","Sessene","Sindia"])
        #arrondissement_data = gdf[(gdf['NAME_1'] == 'Thiès') & (gdf['NAME_3'] == selected_arrondissement)]
  
       
##################################################################################################################################
##############################################Calcul des taux de reproduction de base############################################
#################################################################################################################################
        def equations(x):
            return [
                (r_1 * (1 - epsilon) * x[0] + r_2 * epsilon * x[1]) * (1 - (x[0] + x[1]) / K_R) - (d_1+b_12+b_13) * x[0],
                (r_1 * epsilon * x[0] + r_2 * (1 - epsilon) * x[1]) * (1 - (x[0] + x[1]) / K_R) - (d_2+b_12+b_13) * x[1]
            ]

        # Valeurs initiales
        x01 = [500, 300]

        # Résolution du système d'équations
        x1 = fsolve(equations, x01)

        # Affichage des r?sultats
        S_1 = x1[0]
        S_2 = x1[1]
        b = (1 - np.exp(-alpha * (S_1 + S_2))) / (S_1 + S_2)
        k_1 = ((beta_1 * S_1) * (d_1 + delta_1) * K_A) / ((1 - np.exp(-alpha * (S_1 + S_2)) + d_F+b_12+b_13) * (d_1 + delta_1 + gamma_1+a_12+a_13))
        k_2 = ((beta_3 * S_2) * (d_2 + delta_2) * K_A) / ((1 - np.exp(-alpha * (S_1 + S_2)) + d_F+b_12+b_13) * (d_2 + delta_2 + gamma_2+a_12+a_13))

        R0_1 = np.sqrt(b * max(k_1, k_2))
        #print("la valeur du R0 est : ", R0_1)

        #zone2
        def equations(x):
            return [
                (r2_1 * (1 - epsilon) * x[0] + r2_2 * epsilon * x[1]) * (1 - (x[0] + x[1]) / K2_R) - (d2_1+b_21+b_23) * x[0],
                (r2_1 * epsilon * x[0] + r2_2 * (1 - epsilon) * x[1]) * (1 - (x[0] + x[1]) / K2_R) - (d2_2+b_21+b_23) * x[1]
            ]

        # Valeurs initiales
        x02 = [500, 300]

        # Résolution du système d'équations
        x2 = fsolve(equations, x02)

        # Affichage des r?sultats
        S_1 = x2[0]
        S_2 = x2[1]
        b = (1 - np.exp(-alpha * (S_1 + S_2))) / (S_1 + S_2)
        k_1 = ((beta2_1 * S_1) * (d2_1 + delta2_1) * K2_A) / ((1 - np.exp(-alpha2 * (S_1 + S_2)) + d2_F+b_21+b_23) * (d2_1 + delta2_1 + gamma2_1+a_21+a_23))
        k_2 = ((beta2_3 * S_2) * (d2_2 + delta2_2) * K2_A) / ((1 - np.exp(-alpha2 * (S_1 + S_2)) + d2_F+b_21+b_23) * (d2_2 + delta2_2 + gamma2_2+a_21+a_23))

        R0_2 = np.sqrt(b * max(k_1, k_2))
        #print("la valeur du R0 est : ", R0_2)

        #zone3

        def equations(x):
            return [
                (r3_1 * (1 - epsilon) * x[0] + r3_2 * epsilon * x[1]) * (1 - (x[0] + x[1]) / K3_R) - (d3_1+b_31+b_32) * x[0],
                (r3_1 * epsilon * x[0] + r3_2 * (1 - epsilon) * x[1]) * (1 - (x[0] + x[1]) / K3_R) - (d3_2+b_31+b_32) * x[1]
            ]

        # Valeurs initiales
        x03 = [500, 300]

        # Résolution du système d'équations
        x3 = fsolve(equations, x03)

        # Affichage des r?sultats
        S_1 = x3[0]
        S_2 = x3[1]
        b = (1 - np.exp(-alpha3 * (S_1 + S_2))) / (S_1 + S_2)
        k_1 = ((beta3_1 * S_1) * (d3_1 + delta3_1) * K3_A) / ((1 - np.exp(-alpha3 * (S_1 + S_2)) + d3_F+b_31+b_32) * (d3_1 + delta3_1 + gamma3_1+a_31+a_32))
        k_2 = ((beta3_3 * S_2) * (d3_2 + delta3_2) * K3_A) / ((1 - np.exp(-alpha3 * (S_1 + S_2)) + d3_F+b_31+b_32) * (d3_2 + delta3_2 + gamma3_2+a_31+a_32))

        R0_3 = np.sqrt(b * max(k_1, k_2))
        #print("la valeur du R0 est : ", R0_3)
        
        #####################################################################################################################
        #####################################################################################################################
        ########## Création d'un DataFrame à partir des données de la simulation (simulation fictive)   #####################
        #####################################################################################################################
        #####################################################################################################################

        @st.cache_data
        def convert_df(df):
             # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')


        ################# ZONE 1 ##############################      
        results_df1 = pd.DataFrame({'Temps': t, 'S1': u[:, 0], 'S2': u[:,1], 'I1':u[:,2],'I2': u[:, 3],'A1': u[:,4], 'A2': u[:,5], 'L1':u[:,6] , 'L2' : u[:,7]})
        st.write('Les données de simulation de {}'.format(selected_arrondissements[0]))
        # Sauvegarder les paramètres dans un fichier CSV
        #results_df.to_csv("dataframe1.csv", index=False) 
        st.write(results_df1)  
        csv1 = convert_df(results_df1)
        st.download_button(
             label="Télécharger le tableau",
             data=csv1,
             file_name='df_zone1.csv',
             mime='text/csv',
         )
       
        ################# ZONE 2 ##############################      
        results_df2 = pd.DataFrame({'Temps': t, 'S1': u[:, 8], 'S2': u[:,9], 'I1':u[:,10],'I2': u[:,11],'A1': u[:,12], 'A2': u[:,13], 'L1':u[:,14] , 'L2' : u[:,15]})
        st.write('Les données de simulation de {}'.format(selected_arrondissements[1]))
        # Sauvegarder les paramètres dans un fichier CSV
        #results_df.to_csv("dataframe1.csv", index=False) 
        st.write(results_df2)  
        csv2 = convert_df(results_df2)
        st.download_button(
             label="Télécharger le tableau",
             data=csv2,
             file_name='df_zone2.csv',
             mime='text/csv',
         )
 

        ################# ZONE 3 ##############################      
        results_df3 = pd.DataFrame({'Temps': t, 'S1': u[:,16], 'S2': u[:,17], 'I1':u[:,18],'I2': u[:,19],'A1': u[:,20], 'A2': u[:,21], 'L1':u[:,22] , 'L2' : u[:,23]})
        st.write('Les données de simulation de {}'.format(selected_arrondissements[2]))
        # Sauvegarder les paramètres dans un fichier CSV
        #results_df.to_csv("dataframe1.csv", index=False) 
        st.write(results_df3)  
        csv3 = convert_df(results_df3)
        st.download_button(
             label="Télécharger le tableau",
             data=csv3,
             file_name='df_zone3.csv',
             mime='text/csv',
         )

            ##########################################################################################################
            ##########################################################################################################
            ##############################  Affichage des graphiques   ###############################################
            ##########################################################################################################
            ##########################################################################################################




        st.subheader("affichage des graphiques")
        #affichage des graphiques

        # Dataframes for visualization
        df1 = pd.DataFrame({'t': t, 'S_1': u[:, 0], 'S_2': u[:, 1], 'I_1': u[:, 2], 'I_2': u[:, 3]})
        df2 = pd.DataFrame({'t': t, 'S_1': u[:, 8], 'S_2': u[:, 9], 'I_1': u[:, 10], 'I_2': u[:, 11]})
        df3 = pd.DataFrame({'t': t, 'A_1': u[:, 4], 'A_2': u[:, 5], 'L_1': u[:, 6], 'L_2': u[:, 7]})
        df4 = pd.DataFrame({'t': t, 'A_1': u[:, 12], 'A_2': u[:, 13], 'L_1': u[:, 14], 'L_2': u[:, 15]})
        df5 = pd.DataFrame({'t': t, 'S_1': u[:, 16], 'S_2': u[:, 17], 'I_1': u[:, 18], 'I_2': u[:,19]})
        df6 = pd.DataFrame({'t': t, 'A_1': u[:, 20], 'A_2': u[:, 21], 'L_1': u[:, 22], 'L_2': u[:, 23]})
        ####################################
        ####################################
        # Créez une fonction pour convertir les tracés Matplotlib en tracés Plotly
        def matplotlib_to_plotly(fig, fig_type=go.Figure):
            return fig_type(go.Figure(fig))

        # Créez une fonction pour afficher les graphiques dans Streamlit
        def show_plots():
            # Create subplots using Plotly
            fig = make_subplots(rows=3, cols=2, subplot_titles=[
            'Evol. Rongeurs de {}: R0={:.2f}'.format(selected_arrondissements[0], R0_1),
            'Evol. Puces de {}: R0={:.2f}'.format(selected_arrondissements[0], R0_1),
            'Evol. Rongeurs de {}: R0={:.2f}'.format(selected_arrondissements[1], R0_2),
            'Evol. Puces de {}: R0={:.2f}'.format(selected_arrondissements[1], R0_2),
            'Evol. Rongeurs de {}: R0={:.2f}'.format(selected_arrondissements[2], R0_3),
            'Evol.  Puces de {}: R0={:.2f}'.format(selected_arrondissements[2], R0_3)
            ], shared_xaxes=True)

            # Ajoutez les tracés à la figure
            fig.add_trace(go.Scatter(x=df1['t'], y=df1['S_1'], mode='lines', name='S_1',line=dict(color='blue')),  row=1, col=1)
            fig.add_trace(go.Scatter(x=df1['t'], y=df1['S_2'], mode='lines', name='S_2',line=dict(color='green')),  row=1, col=1)
            fig.add_trace(go.Scatter(x=df1['t'], y=df1['I_1'], mode='lines', name='I_1',line=dict(color='red')),  row=1, col=1)
            fig.add_trace(go.Scatter(x=df1['t'], y=df1['I_2'], mode='lines', name='I_2',line=dict(color='black')),  row=1, col=1)

            # Ajoutez les tracés à la figure
            fig.add_trace(go.Scatter(x=df3['t'], y=df3['A_1'], mode='lines', name='A_1',line=dict(color='blue')),  row=1, col=2)
            fig.add_trace(go.Scatter(x=df3['t'], y=df3['A_2'], mode='lines', name='A_2',line=dict(color='green')),  row=1, col=2)
            fig.add_trace(go.Scatter(x=df3['t'], y=df3['L_1'], mode='lines', name='L_1',line=dict(color='red')),  row=1, col=2)
            fig.add_trace(go.Scatter(x=df3['t'], y=df3['L_2'], mode='lines', name='L_2',line=dict(color='black')),  row=1, col=2)

            # Ajoutez les tracés à la figure
            fig.add_trace(go.Scatter(x=df2['t'], y=df2['S_1'], mode='lines', name='S_1',line=dict(color='blue')),  row=2, col=1)
            fig.add_trace(go.Scatter(x=df2['t'], y=df2['S_2'], mode='lines', name='S_2',line=dict(color='green')),  row=2, col=1)
            fig.add_trace(go.Scatter(x=df2['t'], y=df2['I_1'], mode='lines', name='I_1',line=dict(color='red')),  row=2, col=1)
            fig.add_trace(go.Scatter(x=df2['t'], y=df2['I_2'], mode='lines', name='I_2',line=dict(color='black')),  row=2, col=1)

            # Ajoutez les tracés à la figure
            fig.add_trace(go.Scatter(x=df4['t'], y=df4['A_1'], mode='lines', name='A_1',line=dict(color='blue')),  row=2, col=2)
            fig.add_trace(go.Scatter(x=df4['t'], y=df4['A_2'], mode='lines', name='A_2',line=dict(color='green')),  row=2, col=2)
            fig.add_trace(go.Scatter(x=df4['t'], y=df4['L_1'], mode='lines', name='L_1',line=dict(color='red')),  row=2, col=2)
            fig.add_trace(go.Scatter(x=df4['t'], y=df4['L_2'], mode='lines', name='L_2',line=dict(color='black')),  row=2, col=2)

            # Ajoutez les tracés à la figure
            fig.add_trace(go.Scatter(x=df5['t'], y=df5['S_1'], mode='lines', name='S_1',line=dict(color='blue')),  row=3, col=1)
            fig.add_trace(go.Scatter(x=df5['t'], y=df5['S_2'], mode='lines', name='S_2',line=dict(color='green')),  row=3, col=1)
            fig.add_trace(go.Scatter(x=df5['t'], y=df5['I_1'], mode='lines', name='I_1',line=dict(color='red')),  row=3, col=1)
            fig.add_trace(go.Scatter(x=df5['t'], y=df5['I_2'], mode='lines', name='I_2',line=dict(color='black')),  row=3, col=1)
 
            # Ajoutez les tracés à la figure
            fig.add_trace(go.Scatter(x=df6['t'], y=df6['A_1'], mode='lines', name='A_1',line=dict(color='blue')),   row=3, col=2)
            fig.add_trace(go.Scatter(x=df6['t'], y=df6['A_2'], mode='lines', name='A_2',line=dict(color='green')),  row=3, col=2)
            fig.add_trace(go.Scatter(x=df6['t'], y=df6['L_1'], mode='lines', name='L_1',line=dict(color='red')),    row=3, col=2)
            fig.add_trace(go.Scatter(x=df6['t'], y=df6['L_2'], mode='lines', name='L_2',line=dict(color='black')),  row=3, col=2)

            # Mise à jour de la mise en page
            fig.update_layout(
            height=900,
            showlegend=False,
            legend=dict(x=1.5, y=1.15, traceorder='normal', orientation='h')  # Position de la légende
            #title_text="Affichage des graphiques", 
            )
            #fig.layout.update(xaxis_rangeslider_visible=True, xaxis =dict(title = 'time(day)'), yaxis = dict(title = "Individus"))
           
            

            # Affichage du graphique dans Streamlit
            st.plotly_chart(matplotlib_to_plotly(fig, go.Figure))

        # Appel de la fonction pour afficher les graphiques
        show_plots()   

        ##############################################################################################################################
        ##############################################################################################################################
        #################################   Geolocalisation dans la carte de Thiès    ################################################
        ##############################################################################################################################
        ##############################################################################################################################

 
        # Création d'un DataFrame à partir des données de la simulation (simulation fictive)

        #selected_arrondissements = ['arr1', 'arr2', 'arr3']
        # Chargement du fichier Shapefile avec GeoPandas
        @st.cache_data
        def load_shapefile(shapefile_path):
            return gpd.read_file(shapefile_path)

        shapefile_path = "SEN_adm3.shp"
        gdf = load_shapefile(shapefile_path)

        # Liste des arrondissements uniques dans le jeu de données de Thies
        thies_arrondissements = gdf[gdf['NAME_1'] == 'Thiès']['NAME_3'].unique()
        df = pd.DataFrame({
            'temps': t,
            f'{selected_arrondissements[0]}': u[:, 2] + u[:, 3],
            f'{selected_arrondissements[1]}': u[:, 10] + u[:, 11],
            f'{selected_arrondissements[2]}': u[:, 18] + u[:, 19],
        })

        #st.write("Nombres d\' infection de rongeurs pour chaque zone en fonction du temps")
        #st.write(df)

        # Chargement du fichier Shapefile avec GeoPandas
 

 
        # Convertir le DataFrame en un format adapté à Plotly Express
        df_melted = df.melt(id_vars=["temps"], var_name="NAME_3", value_name="Infections")

        # Fusionner les données d'infection avec les données géographiques des communes
        gdf_merged = gdf[gdf['NAME_3'].isin(thies_arrondissements)].merge(df_melted, left_on='NAME_3', right_on='NAME_3')

        # Créer la carte choroplèthe animée
        fig = px.choropleth_mapbox(
            gdf_merged,
            geojson=gdf_merged.geometry,
            locations=gdf_merged.index,
            color='Infections',
            animation_frame='temps',
            color_continuous_scale='reds',
            mapbox_style="open-street-map",
            title='Nombres de rongeurs infectés par arrondissement au fil du temps',
            labels={'Infections': 'Nombre d\'infections'},
            center={"lat": gdf_merged['geometry'].centroid.y.mean(), "lon": gdf_merged['geometry'].centroid.x.mean()}
        )

        # Afficher la carte
        #fig.show() 
        st.plotly_chart(fig)
        ############################## Pour les puces libre infectueuses ######################
        df1 = pd.DataFrame({
            'temps': t,
            f'{selected_arrondissements[0]}': results_df1['I1'] + results_df1['I2'],
            f'{selected_arrondissements[1]}': results_df2['I1'] + results_df2['I2'],
            f'{selected_arrondissements[2]}': results_df3['I1'] + results_df3['I2']
        })

        #st.write("Nombres de puces libres infectueuses par arrondissement au fil du temps")
        #st.write(df1)

        # Chargement du fichier Shapefile avec GeoPandas
 

 
        # Convertir le DataFrame en un format adapté à Plotly Express
        df_melted = df1.melt(id_vars=["temps"], var_name="NAME_3", value_name="Infections")

        # Fusionner les données d'infection avec les données géographiques des communes
        gdf_merged = gdf[gdf['NAME_3'].isin(thies_arrondissements)].merge(df_melted, left_on='NAME_3', right_on='NAME_3')

        # Créer la carte choroplèthe animée
        fig = px.choropleth_mapbox(
            gdf_merged,
            geojson=gdf_merged.geometry,
            locations=gdf_merged.index,
            color='Infections',
            animation_frame='temps',
            color_continuous_scale='reds',
            mapbox_style="open-street-map",
            title='Nombres de puces libres infectueuses par arrondissement au fil du temps',
            labels={'Infections': 'Nombre de puces infectueuses'},
            center={"lat": gdf_merged['geometry'].centroid.y.mean(), "lon": gdf_merged['geometry'].centroid.x.mean()}
        )

        # Afficher la carte
        #fig.show() 
        st.plotly_chart(fig)

    ########################################### Fin de la fonction prague_model ############################################
        return t , u 
    ########################################################################################################################
    ######################################   Interface utilisateur Streamlit (Les Paramettre)  #############################
    ########################################################################################################################
    #st.title('Modèle de transmission de la peste entre rongeurs et puces')

    # Widgets d'entrée
    

    #Initial1 = [500, 0, 50, 70, 10, 20, 60, 80]
    #Initial2 = [ 0, 500, 0, 0, 0, 0, 0, 0]
    temps_final = st.sidebar.slider("temps_final", 0.0, 1000.0, 10.0)
    temps_final = st.sidebar.number_input("temps_final", min_value=0.0, max_value=1000.0, value=temps_final)
    # Champ de saisie pour les valeurs initiales
    st.sidebar.subheader(" Population initiale")
    Initial1 = st.sidebar.text_input('Population initiale zone 1 (S1,S2,I1,I2,A1,A2,L1,L2)', value="500,0,50,70,10,20,60,80")
    Initial1 = [float(val.strip()) for val in Initial1.split(',')]

    Initial2 = st.sidebar.text_input('Population initiale zone 2 (S1,S2,I1,I2,A1,A2,L1,L2)', value=" 0,500,0,0,0,0,0,0")
    Initial2 = [float(val.strip()) for val in Initial2.split(',')] 

    Initial3 = st.sidebar.text_input('Population initiale zone 3 (S1,S2,I1,I2,A1,A2,L1,L2)', value=" 300,0,0,0,0,0,0,0")
    Initial3 = [float(val.strip()) for val in Initial3.split(',')] 
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


    st.sidebar.subheader("les paramettre pour la zone 3")

    st.sidebar.write("les taux de reproduction des rongeurs")
    r3_1 = st.sidebar.slider("r3_1", 0.0, 1.0, 0.9)
    r3_1 = st.sidebar.number_input("r3_1", min_value=0.0, max_value=1.0, value=r3_1)
    r3_2 = st.sidebar.slider("r3_2", 0.0, 1.0, 0.79)
    r3_2 = st.sidebar.number_input("r3_2", min_value=0.0, max_value=1.0, value=r3_2)
    st.sidebar.write("la capacité limite des rongeurs")
    K3_R = st.sidebar.slider("K3_R", 0, 2000, 1000)
    K3_R = st.sidebar.number_input("K3_R", min_value=0, max_value=2000, value=K3_R)

    alpha3 = st.sidebar.slider("alpha3", 0.0, 1.0, 0.7)
    alpha3 = st.sidebar.number_input("alpha3", min_value=0.0, max_value=1.0, value=alpha3)

    st.sidebar.write("les paramettres de transmision")
    beta3_1 = st.sidebar.slider("beta3_1", 0.0, 1.0, 0.2)
    beta3_1 = st.sidebar.number_input("beta3_1", min_value=0.0, max_value=1.0, value=beta3_1)

    beta3_2 = st.sidebar.slider("beta3_2", 0.0, 1.0, 0.2)
    beta3_2 = st.sidebar.number_input("beta3_2", min_value=0.0, max_value=1.0, value=beta3_2)

    beta3_3 = st.sidebar.slider("beta3_3", 0.0, 1.0, 0.2)
    beta3_3 = st.sidebar.number_input("beta3_3", min_value=0.0, max_value=1.0, value=beta3_3)



    beta3_4 = st.sidebar.slider("beta3_4", 0.0, 1.0, 0.4)
    beta3_4 = st.sidebar.number_input("beta3_4", min_value=0.0, max_value=1.0, value=beta3_4)

    st.sidebar.write("les taux de mortalités naturels des rongeurs")

    d3_1 = st.sidebar.slider("d3_1", 0.0, 1.0, 0.04)
    d3_1 = st.sidebar.number_input("d3_1", min_value=0.0, max_value=1.0, value=d3_1)

    d3_2 = st.sidebar.slider("d3_2", 0.0, 1.0, 0.035)
    d3_2 = st.sidebar.number_input("d3_1", min_value=0.0, max_value=1.0, value=d3_2)
    st.sidebar.write("les taux de guerrisons des rongeurs")
    gamma3_1 = st.sidebar.slider("gamma3_1", 0.0, 1.0, 0.7)
    gamma3_1 = st.sidebar.number_input("gamma3_1", min_value=0.0, max_value=1.0, value=gamma3_1)

    gamma3_2 = st.sidebar.slider("gamma3_2", 0.0, 1.0, 0.7)
    gamma3_2 = st.sidebar.number_input("gamma3_2", min_value=0.0, max_value=1.0, value=gamma3_2)

    delta3_1 = st.sidebar.slider("delta3_1", 0.0, 1.0, 0.2)
    delta3_1 = st.sidebar.number_input("delta3_1", min_value=0.0, max_value=1.0, value=delta3_1)

    st.sidebar.write("les taux de mortalité du à la maladie")
    delta3_2 = st.sidebar.slider("delta3_2", 0.0, 1.0, 0.2)
    delta3_2 = st.sidebar.number_input("delta3_2", min_value=0.0, max_value=1.0, value=delta2_2)

    st.sidebar.write("les taux de reproduction des puces attachées aux rongeurs ")
    ra3_1 = st.sidebar.slider("ra3_1", 0.0, 1.0, 0.4)
    ra3_1 = st.sidebar.number_input("ra3_1", min_value=0.0, max_value=1.0, value=ra3_1)

    ra3_2 = st.sidebar.slider("ra3_2", 0.0, 1.0, 0.5)
    ra3_2 = st.sidebar.number_input("ra3_2", min_value=0.0, max_value=1.0, value=ra3_2)

    st.sidebar.write("la capacité limite des puces attaché aux rongeurs")
    K3_A = st.sidebar.slider("K3_A", 0, 100, 50)
    K3_A = st.sidebar.number_input("K3_A", min_value=0, max_value=100, value=K3_A)

    st.sidebar.write("le taux de mortalité des puces libres")
    d3_F = st.sidebar.slider("d3_F", 0.0, 1.0, 0.8)
    d3_F = st.sidebar.number_input("d3_F", min_value=0.0, max_value=1.0, value=d3_F)

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

    a_31 = st.sidebar.slider("a_31", 0.0, 0.01, 0.0008)
    a_31 = st.sidebar.number_input("a_31", min_value=0.0, max_value=1.0, value=a_31)

    a_23 = st.sidebar.slider("a_23", 0.0, 0.01, 0.0008)
    a_23 = st.sidebar.number_input("a_23", min_value=0.0, max_value=1.0, value=a_23)

    a_13 = st.sidebar.slider("a_13", 0.0, 0.01, 0.0008)
    a_13 = st.sidebar.number_input("a_13", min_value=0.0, max_value=1.0, value=a_13)

    a_32 = st.sidebar.slider("a_32", 0.0, 0.01, 0.0008)
    a_32 = st.sidebar.number_input("a_32", min_value=0.0, max_value=1.0, value=a_32)

    b_12 = st.sidebar.slider("b_12", 0.0, 1.00, 0.02)
    b_12 = st.sidebar.number_input("b_12", min_value=0.0, max_value=1.0, value=b_12)

    b_21 = st.sidebar.slider("b_21", 0.0, 0.001, 0.0004)
    b_21 = st.sidebar.number_input("b_21", min_value=0.0, max_value=1.0, value=b_21)

    b_13 = st.sidebar.slider("b_13", 0.0, 1.00, 0.02)
    b_13 = st.sidebar.number_input("b_13", min_value=0.0, max_value=1.0, value=b_13)

    b_23 = st.sidebar.slider("b_23", 0.0, 1.00, 0.02)
    b_23 = st.sidebar.number_input("b_23", min_value=0.0, max_value=1.0, value=b_23)

    b_32 = st.sidebar.slider("b_32", 0.0, 1.00, 0.02)
    b_32 = st.sidebar.number_input("b_32", min_value=0.0, max_value=1.0, value=b_32)

    b_31 = st.sidebar.slider("b_31", 0.0, 1.00, 0.02)
    b_31 = st.sidebar.number_input("b_31", min_value=0.0, max_value=1.0, value=b_31)



    # Bouton pour exécuter le modèle
    #if st.button('Exécuter le modèle'):
    t, u = model_prague(temps_final, Initial1,Initial2,Initial3, epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2,
                     delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1,
                     beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
                   beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32)

    # Création d'un DataFrame à partir des données de la simulation
    #results_df = pd.DataFrame({'Temps': t, 'Infectés Zone 1': u[:, 2], 'Infectés Zone 2': u[:, 6]})
    # Afficher le DataFrame
    #st.write('Résultats du modèle :')
    #st.dataframe(results_df)

################################################################################################################################
################################################ Mémoire ################################################################
################################################################################################################################    

elif choose == "rapport":
            
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Rapport</p>', unsafe_allow_html=True)
    #with st.expander(label="Veuillez cliquer pour déplier/replier"):
        

    st.markdown("""
        Veuilez cliquer sur le lien ci-dessous pour accéder à notre rapport de stage
         
        * **Rapport:** [https://github.com](https://github.com/AYLY92/memoire/tree/main/Rapport)
        
        """)
            

################################################################################################################################
################################################ CODE SOURCE ##########################################################################
################################################################################################################################

elif choose == "Code source":
            
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Code source</p>', unsafe_allow_html=True)
    st.markdown("""
         Veuillez cliquer sur le lien ci-dessous pour accéder au code source.

        * **Code source:** [https://github.com](https://github.com/AYLY92/memoire/tree/main/code%20source)
        """)
    
         
    
    
    
