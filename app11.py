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
from scipy.optimize import fsolve
 

# sudo sysctl fs.inotify.max_user_watches=1000000

st.set_option('deprecation.showPyplotGlobalUse', False)


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
 
    
def model_prague(Initial1,Initial2,Initial3, epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2,
                 delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1,
                 beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
               beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32):
    # Appel du system_edo pour la résolution
    t = np.arange(0, 100, 1)
    Initial=Initial1+Initial2+Initial3
    
    u = odeint(system_edo, Initial, t, args=(epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2,
                                              gamma_1, gamma_2, delta_1, delta_2,  ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1,
                 beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
               beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32))

   ## return t, u    
    
    
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
    
    
    
    
    
    st.subheader("affichage des graphiques")
    #affichage des graphiques

    # Dataframes for visualization
    df1 = pd.DataFrame({'t': t, 'S_1': u[:, 0], 'S_2': u[:, 1], 'I_1': u[:, 2], 'I_2': u[:, 3]})
    df2 = pd.DataFrame({'t': t, 'S_1': u[:, 8], 'S_2': u[:, 9], 'I_1': u[:, 10], 'I_2': u[:, 11]})
    df3 = pd.DataFrame({'t': t, 'A_1': u[:, 4], 'A_2': u[:, 5], 'L_1': u[:, 6], 'L_2': u[:, 7]})
    df4 = pd.DataFrame({'t': t, 'A_1': u[:, 12], 'A_2': u[:, 13], 'L_1': u[:, 14], 'L_2': u[:, 15]})
    df5 = pd.DataFrame({'t': t, 'S_1': u[:, 16], 'S_2': u[:, 17], 'I_1': u[:, 18], 'I_2': u[:,19]})
    df6 = pd.DataFrame({'t': t, 'A_1': u[:, 20], 'A_2': u[:, 21], 'L_1': u[:, 22], 'L_2': u[:, 23]})
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    # Augmenter l'espace vertical entre les sous-graphiques
    plt.subplots_adjust(hspace=0.3)
    # Plot p1
    axs[0, 0].plot(df1['t'], df1['S_1'], color='blue',label='S_1')
    axs[0, 0].plot(df1['t'], df1['S_2'], color='green',label='S_2')
    axs[0, 0].plot(df1['t'], df1['I_1'], color='red',label='I_1')
    axs[0, 0].plot(df1['t'], df1['I_2'], color='black',label='I_1')
    axs[0, 0].set_title('Evolution des rongeurs dans la zone 1 avec R0={:.2f}'.format(R0_1))
    axs[0, 0].set_xlabel('time (jours)')
    axs[0, 0].set_ylabel('Nombre de rongeurs')
    axs[0, 0].legend()

    # Plot p2
    axs[1, 0].plot(df2['t'], df2['S_1'],'-', color='blue',label='S_1')
    axs[1, 0].plot(df2['t'], df2['S_2'], '-',color='green',label='S_2')
    axs[1, 0].plot(df2['t'], df2['I_1'],'-', color='red',label='I_1')
    axs[1, 0].plot(df2['t'], df2['I_2'],'-', color='black',label='I_2')
    axs[1, 0].set_title('Evolution des rongeurs dans la zone 2 avec R0={:.2f}'.format(R0_2))
    axs[1, 0].set_xlabel('time (jours)')
    axs[1, 0].set_ylabel('Nombres de rongeurs')
    axs[1, 0].legend()

    # Plot p3
    axs[0, 1].plot(df3['t'], df3['A_1'],'--',color='blue',label='A_1')
    axs[0, 1].plot(df3['t'], df3['A_2'],'--',color='green',label='A_2')
    axs[0, 1].plot(df3['t'], df3['L_1'],'--',color='red',label='L_1')
    axs[0, 1].plot(df3['t'], df3['L_2'],'--',color='black',label='L_2')
    axs[0, 1].set_title('Evolution des puces dans la zone 1 avec R0={:.2f}'.format(R0_1))
    axs[0, 1].set_xlabel('time (jours)')
    axs[0, 1].set_ylabel('Nombres de puces')
    axs[0, 1].legend()
    # Plot p4
    axs[1, 1].plot(df4['t'], df4['A_1'],'--', color='blue',label='A_1')
    axs[1, 1].plot(df4['t'], df4['A_2'],'--', color='green',label='A_2')
    axs[1, 1].plot(df4['t'], df4['L_1'],'--', color='red',label='L_1')
    axs[1, 1].plot(df4['t'], df4['L_2'],'--', color='black',label='L_2')
    axs[1, 1].set_title('Evolution des puces dans la zone 2 avec R0={:.2f}'.format(R0_2))
    axs[1, 1].set_xlabel('time (jours)')
    axs[1, 1].set_ylabel('Nombre de puces')
    axs[1, 1].legend()
    # Plot p5
    axs[2, 0].plot(df5['t'], df5['S_1'], color='blue',label='S_1')
    axs[2, 0].plot(df5['t'], df5['S_2'], color='green',label='S_2')
    axs[2, 0].plot(df5['t'], df5['I_1'], color='red',label='I_1')
    axs[2, 0].plot(df5['t'], df5['I_2'], color='black',label='I_2')
    axs[2, 0].set_title('Evolution des rongeurs dans la zone 3 avec R0={:.2f}'.format(R0_3))
    axs[2, 0].set_xlabel('time (jours)')
    axs[2, 0].set_ylabel('nombres des puces')
    axs[2, 0].legend()
    # Plot p6
    axs[2, 1].plot(df6['t'], df6['A_1'],'--', color='blue',label='A_1')
    axs[2, 1].plot(df6['t'], df6['A_2'],'--', color='green',label='A_2')
    axs[2, 1].plot(df6['t'], df6['L_1'],'--', color='red',label='L_1')
    axs[2, 1].plot(df6['t'], df6['L_2'],'--', color='black',label='L_2')
    axs[2, 1].set_title('Evolution des puces dans la zone 3 avec R0={:.2f}'.format(R0_3))
    axs[2, 1].set_xlabel('time (jours)')
    axs[2, 1].set_ylabel('Nombres des puces')
    axs[2, 1].legend()
    # Adjust layout
    plt.tight_layout()

    # Show the plots
    st.pyplot()
    
    # Création d'un DataFrame à partir des données de la simulation (simulation fictive)
 
    results_df = pd.DataFrame({'Temps': t, 'Infectés I1 Zone1': u[:, 2], 'Infectés I2 Zone1': u[:, 3], 'Infectés I1 Zone2':u[:, 10],'Infectés I2 Zone2': u[:, 11],'Infectés I1 Zone3': u[:,18], 'Infectés I2 Zone3': u[:,19]})
    st.write(" affichage du dataframe")
    # Sauvegarder les paramètres dans un fichier CSV
    results_df.to_csv("dataframe1.csv", index=False) 
    st.write(results_df)
     
    infectes_zone_1 = results_df['Infectés I1 Zone1']+results_df['Infectés I2 Zone1']
    infectes_zone_2 = results_df['Infectés I1 Zone2']+results_df['Infectés I2 Zone2']
    infectes_zone_3 = results_df['Infectés I1 Zone3']+results_df['Infectés I2 Zone3']
 
 
 
    val1=[(infectes_zone_1[j]-min(infectes_zone_1))/(max(infectes_zone_1)-min(infectes_zone_1)) for j in range(len(infectes_zone_1))]
 
    val2=[(infectes_zone_2[j]-min(infectes_zone_2))/(max(infectes_zone_2)-min(infectes_zone_2)) for j in range(len(infectes_zone_2))]
 
    val3=[(infectes_zone_3[j]-min(infectes_zone_3))/(max(infectes_zone_3)-min(infectes_zone_3)) for j in range(len(infectes_zone_3))]
 
    results_df1 = pd.DataFrame({'Temps': t, 'sum_infecté_Zone1': val1, 'sum_infecté_Zone2': val2, 'sum_infecté_Zone3':val3})
    st.write(results_df1)
    results_df1.to_csv("df2.csv",index=False)
    
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
         if idx == 1:
            arrondissement_data.plot(ax=ax, color="#BB0000", legend=True)            
         else :
            arrondissement_data.plot(ax=ax, color="#660000", legend=True)
       
    if selected_arrondissements==[]:
        st.write("veuiller selectionner deux zone à etudier")
  
    # Ajouter une barre de couleur (colorbar) pour indiquer les valeurs d'infections
    cbar = plt.colorbar(sm_bar, ax=ax, format='%d')
    cbar.set_label('Nombres de cas infecés')

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
t, u = model_prague(Initial1,Initial2,Initial3, epsilon, r_1, r_2, K_R, alpha, beta_1, beta_2, beta_3, beta_4, d_1, d_2, gamma_1, gamma_2,
                 delta_1, delta_2, ra_1, ra_2, K_A, d_F, a_12, a_21, b_12, b_21, r2_1, r2_2, K2_R, alpha2, beta2_1,
                 beta2_2, beta2_3, beta2_4, d2_1, d2_2, gamma2_1, gamma2_2, delta2_1, delta2_2, ra2_1, ra2_2, K2_A, d2_F, r3_1, r3_2, K3_R, alpha3, beta3_1, beta3_2, beta3_3,
               beta3_4, d3_1, d3_2, gamma3_1, gamma3_2, delta3_1, delta3_2, ra3_1, ra3_2, K3_A, d3_F,a_13,a_31,a_23,a_32,b_13,b_31,b_23,b_32)
 
# Création d'un DataFrame à partir des données de la simulation
#results_df = pd.DataFrame({'Temps': t, 'Infectés Zone 1': u[:, 2], 'Infectés Zone 2': u[:, 6]})
# Afficher le DataFrame
#st.write('Résultats du modèle :')
#st.dataframe(results_df)


 














 
