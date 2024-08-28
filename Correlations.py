###################################################################################################################################################
#                                                                                                                                                 #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt       #
#   Autor: Peronnik Unverzagt (peronnik.unverzagt@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt   #                                                                                                                                           #
#                                                                                                                                                 #
###################################################################################################################################################

# PYTHON ROUTINE zur Berechnung der Korrelation # 
import os
import sys
import statistics
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    df = pd.read_csv('Kaggle\data.csv')

    df_realistic = pd.read_csv('RealisticModelData.csv')

    df_reduced = df.replace({'Oculus Rift': 1.0, 'HTC Vive': 2.0, 'PlayStation VR': 3.0, 'Male': 4.0, 'Female':5.0, 'Other':6.0})

    df_reduced_realistic = df_realistic.replace({'Oculus Rift': 1.0, 'HTC Vive': 2.0, 'PlayStation VR': 3.0, 'Male': 4.0, 'Female':5.0, 'Other':6.0})

    #df_reduced = df[['Gender']].copy()

    #print(df_reduced.value_counts())

    #print(df_reduced)
    print(df_reduced_realistic.corr())
    print("   ")
    print("   ")
    print(df_reduced.corr())
    print("   ")
    print("   ")
    print(df_reduced.corr(method='spearman'))
    print("   ")
    print("   ")
    print(df_reduced.corr(method='pearson'))

#HTC Vive,42.574083299362634,6,3
#5,51,Male,PlayStation VR,22.452646660410768,4,2
#6,46,Other,Oculus Rift