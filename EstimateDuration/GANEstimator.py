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
    
    df = pd.read_csv('EstimateDuration\ModelData.csv')
    df_real = pd.read_csv('Kaggle\data.csv')       
    UserID = 0
    found_value = 0

    maximal_accuracy = 75.0 # Maximale Genauigkeit in Prozent
        # Open reference file
        

    df_reduced = df['Duration'].copy()
    Duration = int(df_reduced.mean())
    sample = open('GAN-Results.csv', 'a')
    df_reduced = df['Age'].copy()
    AgeMean = df_reduced.mean()
    MotionSickness = 2
    ImmersionLevel = 2
    Age = 23
    VRHeadset = 'HTC Vive'
    Gender = 'Male'
    

    for i in range (0, len(df)):
        
    
        similarity = 0
        data_accuracy = 20.0 # Zulässige Abweichung in Prozent
        
        if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['MotionSickness']-float(MotionSickness))/float(MotionSickness)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['ImmersionLevel'])-float(ImmersionLevel))/float(ImmersionLevel)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

            found_value = 1

            Duration = df.iloc[i]['Duration']
        

        data_accuracy = 10.0 # Zulässige Abweichung in Prozent

        if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['MotionSickness']-float(MotionSickness))/float(MotionSickness)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['ImmersionLevel'])-float(ImmersionLevel))/float(ImmersionLevel)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

            found_value = 1

            Duration = df.iloc[i]['Duration']
        

        data_accuracy = 25.0 # Zulässige Abweichung in Prozent

        if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['MotionSickness']-float(MotionSickness))/float(MotionSickness)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['ImmersionLevel'])-float(ImmersionLevel))/float(ImmersionLevel)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):


            found_value = 1

            Duration = df.iloc[i]['Duration']
            
        data_accuracy = 50.0 # Zulässige Abweichung in Prozent

        if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['MotionSickness']-float(MotionSickness))/float(MotionSickness)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['ImmersionLevel'])-float(ImmersionLevel))/float(ImmersionLevel)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):
    

            found_value = 1

            Duration = df.iloc[i]['Duration']
    
        data_accuracy = 75.0 # Zulässige Abweichung in Prozent

        if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['MotionSickness']-float(MotionSickness))/float(MotionSickness)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['ImmersionLevel'])-float(ImmersionLevel))/float(ImmersionLevel)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

            found_value = 1

            Duration = df.iloc[i]['Duration']
            

        data_accuracy = maximal_accuracy # Zulässige Abweichung in Prozent

        if ((df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['MotionSickness']-float(MotionSickness))/float(MotionSickness)*100.0 < (100.0 - data_accuracy) and (float(df.iloc[i]['ImmersionLevel'])-float(ImmersionLevel))/float(ImmersionLevel)*100.0 < (100.0 - data_accuracy) and df.iloc[i]['Gender'] == Gender and df.iloc[i]['VRHeadset'] == VRHeadset):

            found_value = 1

            Duration = df.iloc[i]['Duration']
            

    durationEstimate = open('EstimateDuration\durationEstimate.txt', 'w')
    df_reduced = df['Duration'].copy()
    maxDuration = df_reduced.max()
    
    print('Duration:',  Duration)
    print(Duration / maxDuration, file = durationEstimate)
    
