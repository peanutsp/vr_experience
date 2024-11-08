###################################################################################################################################################
#                                                                                                                                                 #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt       #
#   Autor: Peronnik Unverzagt (peronnik.unverzagt@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt   #                                                                                                                                           #
#                                                                                                                                                 #
###################################################################################################################################################

# PYTHON ROUTINE zur Modellierung von Daten durch GAN-Netzwerke #

import os
import sys
import statistics
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

def custom_round(value):
    rounded_value = round(value)
    if rounded_value == 0:
        if value > 0:
            return 1
        elif value < 0:
            return -1
    return rounded_value

def Generator(ID, length):

    output_key = [0]*length
    output_key[0] = ID

    for k in range(0, length):

        if (k == 1):

        

            output_key[1] = random.randint(10, 100) # Alter im Bereich von zehr bis Hundert Jahre

        if (k == 2):

            choose_gender = random.randint(0,1)

            if (choose_gender == 0):

                output_key[2] = 'Male' 
            
            if (choose_gender == 1):

                output_key[2] = 'Female' 

        if (k == 3):

            choose_headset_type = random.randint(0,2)

            if (choose_headset_type == 0):

                output_key[3] = 'Oculus Rift' 
            
            if (choose_headset_type == 1):

                output_key[3] = 'HTC Vive' 

            if (choose_headset_type == 2):

                output_key[3] = 'PlayStation VR' 

        if (k == 4):

            
            
            duration_mean = df['Duration'].mean()
            duration_std = df['Duration'].std()

            #output_key[4] = random.gauss(duration_mean,duration_std)
            output_key[4] = random.uniform(df['Duration'].min(),df['Duration'].max())

        if (k == 5):

           

            motionsickness_mean = df['MotionSickness'].mean()
            motionsickness_std = df['MotionSickness'].std()
            
            output_key[5] = random.randint(df['MotionSickness'].min(),df['MotionSickness'].max())
           # output_key[5] = custom_round(random.gauss(motionsickness_mean,motionsickness_std))

        if (k == 6):

           

            immersion_mean = df['ImmersionLevel'].mean()
            immersion_std = df['ImmersionLevel'].std()
            #output_key[6] = custom_round(random.gauss(immersion_mean,immersion_std))
            output_key[6] = random.randint(df['ImmersionLevel'].min(),df['ImmersionLevel'].max())


    return(output_key)


def Differentiator(df, data_1, data_2, data_3, data_4, data_5, data_6, data_7, accuracy):

    similarity_measure = 0 

    if ((df.iloc[j]['Age']-data_2)/data_2*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Duration']-data_5)/data_5*100.0 < (100.0 -  accuracy) and (df.iloc[j]['MotionSickness']-data_6)/data_6*100.0 < (100.0 -  accuracy) and (df.iloc[j]['ImmersionLevel']-data_7)/data_7*100.0 < (100.0 -  accuracy)):

        similarity_measure = 1

    return similarity_measure

if __name__ == "__main__":
    
    
    # Replace 'your_file.csv' with the path to your CSV file
    df = pd.read_csv('C:/Users/Pero/Documents/Python Scripts/Kaggle/data.csv')

    # Code for printing to a file
    sample = open('EstimateDuration\ModelData.csv', 'w')

    # Code for printing to a file
    sample_realistic = open('EstimateDuration\RealisticModelData.csv', 'w')

    data = []
    index = 0
    sample_size = 1000
    number_of_similar_data_maps = 5
    data_dimension = 7
    data_similarity = 90.0

    print(','.join(['UserID', 'Age', 'Gender', 'VRHeadset', 'Duration', 'MotionSickness', 'ImmersionLevel']), file=sample)

    for i in range(1, sample_size):
        print (i)
        writer_var = Generator(i, data_dimension)
        print(f"{index},{writer_var[1]},{writer_var[2]},{writer_var[3]},{writer_var[4]},{writer_var[5]},{writer_var[6]}", file=sample)

    print(','.join(['UserID', 'Age', 'Gender', 'VRHeadset', 'Duration', 'MotionSickness', 'ImmersionLevel']), file=sample_realistic)

    for j in range(0, number_of_similar_data_maps):

        print(j)

        for i in range(1, 1001): #TODO: number of rows of pf

            similarity = 0

            while(1==1):

                writer_var = Generator(i, data_dimension)


        
        # if ((df.iloc[j]['Age']-writer_var[1])/writer_var[1]*100.0 < (100.0 - data_similarity) and (df.iloc[j]['Duration']-writer_var[4])/writer_var[4]*100.0 < (100.0 - data_similarity) and (df.iloc[j]['MotionSickness']-writer_var[5])/writer_var[5]*100.0 < (100.0 - data_similarity) and (df.iloc[j]['ImmersionLevel']-writer_var[6])/writer_var[6]*100.0 < (100.0 - data_similarity)):

                similarity = Differentiator(df, writer_var[0], writer_var[1], writer_var[2], writer_var[3], writer_var[4], writer_var[5], writer_var[6], data_similarity)
                    
                if (similarity == 1):

                    print(f"{index+1},{writer_var[1]},{writer_var[2]},{writer_var[3]},{writer_var[4]},{writer_var[5]},{writer_var[6]}", file=sample_realistic)
                    index = index + 1

                if (similarity == 1): 
                
                    break    


    # Versuche den Unterschied zwischen dem SVM-Modell und einem Modell wie Decision Tree anhand der Ergebnisse und der Modell-Definitionen zu verstehen