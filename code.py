# -*- coding: utf-8 -*-
"""
travaux_recuit_simulé_15_janvier.ipynb
"""

# Les packages utilisés

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import time
import keyboard
import random as r

# pathIn est le chemin pour enregister les images de configurations ontenues
# à chaque itération réuissie

#pathIn = r"D:\TP recuit simulé\test1"
pathIn = r"D:\Master2_SDTS\optimisation\tpRecuit"

# la définition de la configuration optimale de 25 composants électronqiue
 
G = nx.grid_2d_graph(5,5)  

Config_Optimale={(0, 0): np.array([0, 0]),
 (0, 1): np.array([0, 1]),
 (0, 2): np.array([0 , 2]),
 (0, 3): np.array([0, 3 ]),
 (0, 4): np.array([0, 4 ]),
 (1, 0): np.array([ 1, 0]),
 (1, 1): np.array([1, 1]),
 (1, 2): np.array([1, 2]),
 (1, 3): np.array([1, 3]),
 (1, 4): np.array([1, 4]),
 (2, 0): np.array([ 2, 0]),
 (2, 1): np.array([ 2, 1]),
 (2, 2): np.array([ 2, 2]),
 (2, 3): np.array([2, 3]),
 (2, 4): np.array([2, 4]),
 (3, 0): np.array([3, 0]),
 (3, 1): np.array([ 3, 1]),
 (3, 2): np.array([3, 2]),
 (3, 3): np.array([3,3]),
 (3, 4): np.array([3, 4]),
 (4, 0): np.array([ 4, 0]),
 (4, 1): np.array([4, 1]),
 (4, 2): np.array([4, 2]),
 (4, 3): np.array([4, 3]),
 (4, 4): np.array([4, 4])}

# la fonction pour dessiner une configuration choisi en attribuant un titre et un nom

def Dessin_graphe(configuration,titre_graphe,nom_graphe):
  plt.plot()
  plt.title(titre_graphe)
  H = G.to_directed()
  nx.draw(H, configuration, node_color=range(25), node_size=500, with_labels=True, width=1)
  plt.savefig(pathIn+"/{}.jpg".format(nom_graphe))#,dpi=200)
  plt.show()

# Dessiner le montage optimale == configuation optimale

Dessin_graphe(Config_Optimale,'Montage optimale','Montage optimale')

# dictionnaire: associe chaque composant électronique à ses coordonnées dans le montage optimale

import numpy as np
Composant = {}
Composant[1] = np.array([0,0])
Composant[2] = np.array([0,1])
Composant[3] = np.array([0,2])
Composant[4] = np.array([0,3])
Composant[5] = np.array([0,4])
Composant[6] = np.array([1,0])
Composant[7] = np.array([1,1])
Composant[8] = np.array([1,2])
Composant[9] = np.array([1,3])
Composant[10] = np.array([1,4])
Composant[11] = np.array([2,0])
Composant[12] = np.array([2,1])
Composant[13] = np.array([2,2])
Composant[14] = np.array([2,3])
Composant[15] = np.array([2,4])
Composant[16] = np.array([3,0])
Composant[17] = np.array([3,1])
Composant[18] = np.array([3,2])
Composant[19] = np.array([3,3])
Composant[20] = np.array([3,4])
Composant[21] = np.array([4,0])
Composant[22] = np.array([4,1])
Composant[23] = np.array([4,2])
Composant[24] = np.array([4,3])
Composant[25] = np.array([4,4])

# fonction qui calcule la distance entre deux composants C1(Cx1,Cy1) et C2(Cx2,Cy2)
# dans une configuration bien définie

def distance(Cx1,Cy1,Cx2,Cy2,configuration):
    x1=configuration[(Cx1 ,Cy1)][0]
    x2=configuration[(Cx2 ,Cy2)][0]
    y1=configuration[(Cx1 ,Cy1)][1]
    y2=configuration[(Cx2 ,Cy2)][1]
    dis=(abs(x1-x2)+abs(y1-y2))*5
    return dis

# fonction qui calcule la distance entre les voisins descendants de composants C1
# dans une configuration bien définie

def distance_voisins(C1,configuration):
    (Cx1,Cy1)=Composant[C1]
    if (Cx1!=0 and Cy1!=0):
        dis=distance(Cx1,Cy1,Cx1-1,Cy1,configuration)+distance(Cx1,Cy1,Cx1,Cy1-1,configuration)
    elif (Cx1==0 and Cy1!=0):
        dis=distance(Cx1,Cy1,Cx1,Cy1-1,configuration)
    elif (Cx1!=0 and Cy1==0):
        dis=distance(Cx1,Cy1,Cx1-1,Cy1,configuration)
    elif (Cx1==0 and Cy1==0):
        dis=0
        
    return dis

# fonction qui calcule la distance totale de la configuration des composants

def Dist_totale(configuration):
  dist_totale=0
  for i in range(1,26):
    dist_totale=dist_totale+distance_voisins(i,configuration)
    
  return (dist_totale)

# fonction random qui assure de choisir deux composants différents pour l'opération de permutation 

def jeter_de():
    C1=r.randrange(1, 26)
    C2=C1
    while (C1==C2):
        C2=r.randrange(1, 26)
    return (C1,C2)

# fonction qui assure l'opération de permutation dans une configuration

def permut(configuration):
    (C1,C2)=jeter_de()
    (Cx1,Cy1)=Composant[C1]
    (Cx2,Cy2)=Composant[C2]
    (configuration[(Cx1,Cy1)],configuration[(Cx2,Cy2)])=(configuration[(Cx2,Cy2)],configuration[(Cx1,Cy1)])    
    return configuration

# fonction qui assure la génération d'une configuration initiale
# N: le nombre au choix des permutations à effectués
 
def Configuration_depart():
   i=0
   N=200
   CF=Config_Optimale.copy()
   while (i < N):
     config=permut(CF)
     i=i+1
   return(config)

# La génération de configuration de départ
# calculer la distance globale de configuration

depart= Configuration_depart()
Dessin_graphe(depart,'Montage Départ','Montage Départ')
Dist_totale(depart)

# Fonction pour convertir le temps d"exécution en H:M:S

def convert_to_hours(x): 
        
        hh=(x//60)//60
        mm=((x-hh*3600)//60 )
        ss=(x-hh*60)%60
        
        affi="Temps d'exécution: {}H:{}M:{}S ".format(int(hh),int(mm),int(ss))
        return affi

#exemple de convertir 3600 secondes

convert_to_hours(3600)

# fonction pour savoir si le sytème est figé ou pas
# Nombre de paliers de température successifs sans aucun mouvement accepté

def ARRET(ch1,liste):
    occ=0
    for i in liste:
        if (ch1== i):
            occ+=1
    if (occ==len(liste)):
        return True
    else:
        return False

def recuit_simule(Configuration_depart):
    tic = time.perf_counter()
    i=0
    
    T=1000
    alpha= 0.8
    c1=depart.copy()
    liste_config=[]
    liste_config.append(depart)
    d1=Dist_totale(depart)
    liste_T=[]
    liste_T.append(T)
    liste_distance=[]
    liste_distance.append(d1)
    liste_iteration=[]
    liste_iteration.append(0)
    
    
    while ((T>0)and (d1 > 200)):
        MA=0 # nombre de mouvements acceptés
        MT=0 # nombre de mouvements tentés
                        
        
        while((MA <25) and (MT<100) ):
            c2=permut(c1.copy())
            i=i+1
            d2=Dist_totale(c2)
            Delta=d2-d1
            if (Delta <= 0):
                c1=c2.copy()
                liste_config.append(c2)
                d1=Dist_totale(c1)
                MA+=1
                MT+=1
                liste_distance.append(d1)
                liste_iteration.append(i)
                toc = time.perf_counter()
                Dessin_graphe(c1,'Distance= '+str (d1)+'   '+'Permutations éffectuées: '+str(i)+'\n'+convert_to_hours(round((toc - tic),0))+'   '+'Temperature: '+str(round(T,5)),format(i))

            elif ((math.exp((-Delta)/T)) > r.uniform(0,1)):#Loi de Boltzmann
                c1=c2.copy()
                liste_config.append(c2)
                d1=Dist_totale(c1)
                MA+=1
                MT+=1
                liste_distance.append(d1)
                liste_iteration.append(i)
                toc = time.perf_counter()
                Dessin_graphe(c1,'Distance= '+str (d1)+'   '+'Permutations éffectuées: '+str(i)+'\n'+convert_to_hours(round((toc - tic),0))+'   '+'Temperature: '+str(round(T,5)),format(i))

            else:
                MT+=1
            print(MA,MT)
                
        T=alpha*T # procédure de refroidissement
        liste_T.append(T)    
         
        
        # condition système est figé ou non
        # 3 paliers de température successifs sans aucun mouvement accepté 
        
        if ((ARRET(liste_distance[-1],liste_distance[-3:-1])== True) ):
            break
        else:
            pass
            
        
    return (liste_distance,liste_iteration,liste_config,liste_T)

# fonction de recuit simulé
#   input:  configuration initiale
#   output: liste_distance  : la liste de l'évolution de la distance en fonctions des itérations éffecturées
#           liste_iteration : la liste des itérations efféctuées
#           liste_config    : la liste des configurations acceptés en fonctions des itérations éffecturées
#           liste_T         : la liste des températures en fonctions des itérations éffecturées (procédures de refroidissements)

(liste_distance,liste_iteration,liste_config,liste_T)=recuit_simule(depart)

# Figure l'évolution de la distance en fonctions des itérations éffecturées

plt.plot(liste_iteration,liste_distance)
plt.savefig(pathIn+"/{}.jpg".format("Evolution de la distance en fonctions des itérations éffecturées"))

# Figure l'évolution de la Température en fonctions des itérations éffecturées
plt.plot(liste_T)
plt.savefig(pathIn+"/{}.jpg".format("Refroidissement"),dpi=200)