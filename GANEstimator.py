################################################################################################################################################
#                                                                                                                                              #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt    #
#                                                                                                                                              #
################################################################################################################################################

# PYTHON ROUTINE zur Modellierung von Immersionslevels durch GAN-Modelle

import os
import sys
import statistics
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hier soll eine Routine entwickelt werden, welche ein Immersionslevel findet für vogegebene Parameter (UserID, Age, Gender, VRHeadset, Duration)



if __name__ == "__main__":
    
    df = pd.read_csv('RealisticModelData.csv')
    df_real = pd.read_csv('Kaggle\data.csv')       
    UserID = 0
    found_value = 0

    maximal_accuracy = 75.0 # Maximale Genauigkeit in Prozent
        # Open reference file
        

    df_reduced = df['ImmersionLevel'].copy()
    ImmersionLevel = int(df_reduced.mean())
    sample = open('GAN-Results.csv', 'a')
    df_reduced = df['MotionSickness'].copy()
    motion_sickness = df_reduced.mean()
    
    
    for j in range(0, len(df_real)):
        
        VRHeadset = df_real.iloc[j]['VRHeadset']
        Gender = df_real.iloc[j]['Gender']
        Age = df_real.iloc[j]['Age']
        Duration = df_real.iloc[j]['Duration']
            
        print(j)
        for i in range (0, len(df)):
            
       
            similarity = 0
            data_accuracy = 20.0 # Zulässige Abweichung in Prozent
            
            if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['Duration'])-float(Duration))/float(Duration)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

                found_value = 1

                ImmersionLevel = df.iloc[i]['ImmersionLevel']
                motion_sickness = df.iloc[i]['MotionSickness']

            data_accuracy = 10.0 # Zulässige Abweichung in Prozent

            if ((found_value) != 1 and (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['Duration'])-float(Duration))/float(Duration)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

                found_value = 1

                ImmersionLevel = df.iloc[i]['ImmersionLevel']
                motion_sickness = df.iloc[i]['MotionSickness']

            data_accuracy = 25.0 # Zulässige Abweichung in Prozent

            if ((found_value) != 1 and (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['Duration'])-float(Duration))/float(Duration)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):


                found_value = 1

                ImmersionLevel = df.iloc[i]['ImmersionLevel']
                motion_sickness = df.iloc[i]['MotionSickness']

            data_accuracy = 50.0 # Zulässige Abweichung in Prozent

            if ((found_value) != 1 and (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['Duration'])-float(Duration))/float(Duration)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):
        

                found_value = 1

                ImmersionLevel = df.iloc[i]['ImmersionLevel']
                motion_sickness = df.iloc[i]['MotionSickness']

            data_accuracy = 75.0 # Zulässige Abweichung in Prozent

            if ((found_value) != 1 and (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['Duration'])-float(Duration))/float(Duration)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

                found_value = 1

                ImmersionLevel = df.iloc[i]['ImmersionLevel']
                motion_sickness = df.iloc[i]['MotionSickness']

            data_accuracy = maximal_accuracy # Zulässige Abweichung in Prozent

            if ((found_value) != 1 and (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['Duration'])-float(Duration))/float(Duration)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

                found_value = 1

                ImmersionLevel = df.iloc[i]['ImmersionLevel']
                motion_sickness = df.iloc[i]['MotionSickness']

    
        print(ImmersionLevel, file=sample)
        
