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





MusicEffects = 'Improve'  # (0 = no effect, 1 = improve , 2 = worse)
Permissions = 'I understand'
Duration = 0.0
StatsClass = "Sometimes"


with open("EstimateDuration\durationEstimate.txt", "r") as file:
    for line in file:
        # Split each line into key and value at ": "
        Duration = line



data = {}
with open("Input.txt", "r") as file:
    for line in file:
        # Split each line into key and value at ": "
        key, value = line.strip().split(": ", 1)
        data[key] = value


# Assign the variables from the data dictionary
Age = int(data["Age"])
Anxiety = int(data["Anxiety Level"])
Depression = int(data["Depression Level"])
Insomnia = int(data["Insomnia Level"])
OCD = int(data["OCD Level"])
Exploratory = data["Exploratory"]
ForeignLanguage = data["Foreign Language"]
FavoriteGenre = data["Favorite Genre"]
Instrumentalist = data["Instrumentalist"]
Composer = data["Composer"]
PrimaryStreamingService = data["Primary Streaming Service"]
WhileWorking = data["While Working"]
MusicEffect = data["Music Effect"]



maximal_accuracy = 95.0 # Maximale Genauigkeit in Prozent

# Open reference file
df = pd.read_csv('MusicExperienceData.csv')

#df_reduced = df[firstParameter].copy()
#age = df_reduced.mean()

counter = 0
for i in range(1, len(df)):
    
    found_value = 0
    similarity = 0
    data_accuracy = 5.0 # Zulässige Abweichung in Prozent
    data_accuracy_time = 80.0

    if (((df.iloc[i]['Hours per day']/df['Hours per day'].max())-float(Duration)) < 0.5 and  (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Anxiety']-float(Anxiety))/float(Anxiety+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Depression']-float(Depression))/float(Depression+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Insomnia'] -float(Insomnia))/float(Insomnia+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['OCD']-float(OCD))/float(OCD+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Music effects'] == MusicEffect )): #  and (df.iloc[i]['Exploratory'] == Exploratory) and (df.iloc[i]['Foreign languages'] == ForeignLanguage) and (df.iloc[i]['Fav genre'] == FavoriteGenre ) and (df.iloc[i]['Instrumentalist'] == Instrumentalist ) and (df.iloc[i]['Composer'] == Composer ) and (df.iloc[i]['Primary streaming service'] == PrimaryStreamingService ) and (df.iloc[i]['While working'] == WhileWorking ) and (df.iloc[i]['Music effects'] == MusicEffect )):

        if ( found_value == 0):
            counter += 1
            found_value = 1
        genre_1 = df.iloc[i]['Frequency [Classical]']
        genre_2 = df.iloc[i]['Frequency [Country]']
        genre_3 = df.iloc[i]['Frequency [EDM]']
        genre_4 = df.iloc[i]['Frequency [Gospel]']
        genre_5 = df.iloc[i]['Frequency [Hip hop]']
        genre_6 = df.iloc[i]['Frequency [Jazz]']
        genre_7 = df.iloc[i]['Frequency [K pop]']
        genre_8 = df.iloc[i]['Frequency [Folk]']
        genre_9 = df.iloc[i]['Frequency [Latin]']
        genre_10 = df.iloc[i]['Frequency [Lofi]']
        genre_11 = df.iloc[i]['Frequency [Metal]']
        genre_12 = df.iloc[i]['Frequency [Pop]']
        genre_13 = df.iloc[i]['Frequency [Rap]']
        genre_14 = df.iloc[i]['Frequency [Rock]']
        genre_15 = df.iloc[i]['Frequency [R&B]']
        genre_16 = df.iloc[i]['Frequency [Video game music]']
        duration = df.iloc[i]['Hours per day']
        bpm = df.iloc[i]['BPM']



    data_accuracy = 10.0 # Zulässige Abweichung in Prozent
    if (((df.iloc[i]['Hours per day']/df['Hours per day'].max())-float(Duration)) < 0.5 and  (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Anxiety']-float(Anxiety))/float(Anxiety+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Depression']-float(Depression))/float(Depression+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Insomnia'] -float(Insomnia))/float(Insomnia+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['OCD']-float(OCD))/float(OCD+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Music effects'] == MusicEffect )): # and (df.iloc[i]['Exploratory'] == Exploratory) and (df.iloc[i]['Foreign languages'] == ForeignLanguage) and (df.iloc[i]['Fav genre'] == FavoriteGenre ) and (df.iloc[i]['Instrumentalist'] == Instrumentalist ) and (df.iloc[i]['Composer'] == Composer ) and (df.iloc[i]['Primary streaming service'] == PrimaryStreamingService ) and (df.iloc[i]['While working'] == WhileWorking ) and (df.iloc[i]['Music effects'] == MusicEffect )):

        if ( found_value == 0):
            counter += 1
            found_value = 1

        genre_1 = df.iloc[i]['Frequency [Classical]']
        genre_2 = df.iloc[i]['Frequency [Country]']
        genre_3 = df.iloc[i]['Frequency [EDM]']
        genre_4 = df.iloc[i]['Frequency [Gospel]']
        genre_5 = df.iloc[i]['Frequency [Hip hop]']
        genre_6 = df.iloc[i]['Frequency [Jazz]']
        genre_7 = df.iloc[i]['Frequency [K pop]']
        genre_8 = df.iloc[i]['Frequency [Folk]']
        genre_9 = df.iloc[i]['Frequency [Latin]']
        genre_10 = df.iloc[i]['Frequency [Lofi]']
        genre_11 = df.iloc[i]['Frequency [Metal]']
        genre_12 = df.iloc[i]['Frequency [Pop]']
        genre_13 = df.iloc[i]['Frequency [Rap]']
        genre_14 = df.iloc[i]['Frequency [Rock]']
        genre_15 = df.iloc[i]['Frequency [R&B]']
        genre_16 = df.iloc[i]['Frequency [Video game music]']
        duration = df.iloc[i]['Hours per day']
        bpm = df.iloc[i]['BPM']
        
        
    data_accuracy = 25.0 # Zulässige Abweichung in Prozent

    if (((df.iloc[i]['Hours per day']/df['Hours per day'].max())-float(Duration)) < 0.5 and  (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Anxiety']-float(Anxiety))/float(Anxiety+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Depression']-float(Depression))/float(Depression+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Insomnia'] -float(Insomnia))/float(Insomnia+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['OCD']-float(OCD))/float(OCD+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Music effects'] == MusicEffect )): # and (df.iloc[i]['Exploratory'] == Exploratory) and (df.iloc[i]['Foreign languages'] == ForeignLanguage) and (df.iloc[i]['Fav genre'] == FavoriteGenre ) and (df.iloc[i]['Instrumentalist'] == Instrumentalist ) and (df.iloc[i]['Composer'] == Composer ) and (df.iloc[i]['Primary streaming service'] == PrimaryStreamingService ) and (df.iloc[i]['While working'] == WhileWorking ) and (df.iloc[i]['Music effects'] == MusicEffect )):
 
        if ( found_value == 0):
            counter += 1
            found_value = 1

        genre_1 = df.iloc[i]['Frequency [Classical]']
        genre_2 = df.iloc[i]['Frequency [Country]']
        genre_3 = df.iloc[i]['Frequency [EDM]']
        genre_4 = df.iloc[i]['Frequency [Gospel]']
        genre_5 = df.iloc[i]['Frequency [Hip hop]']
        genre_6 = df.iloc[i]['Frequency [Jazz]']
        genre_7 = df.iloc[i]['Frequency [K pop]']
        genre_8 = df.iloc[i]['Frequency [Folk]']
        genre_9 = df.iloc[i]['Frequency [Latin]']
        genre_10 = df.iloc[i]['Frequency [Lofi]']
        genre_11 = df.iloc[i]['Frequency [Metal]']
        genre_12 = df.iloc[i]['Frequency [Pop]']
        genre_13 = df.iloc[i]['Frequency [Rap]']
        genre_14 = df.iloc[i]['Frequency [Rock]']
        genre_15 = df.iloc[i]['Frequency [R&B]']
        genre_16 = df.iloc[i]['Frequency [Video game music]']
        duration = df.iloc[i]['Hours per day']
        bpm = df.iloc[i]['BPM']

    data_accuracy = 50.0 # Zulässige Abweichung in Prozent

    if (((df.iloc[i]['Hours per day']/df['Hours per day'].max())-float(Duration)) < 0.5 and  (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Anxiety']-float(Anxiety))/float(Anxiety+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Depression']-float(Depression))/float(Depression+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Insomnia'] -float(Insomnia))/float(Insomnia+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['OCD']-float(OCD))/float(OCD+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Music effects'] == MusicEffect )): # and (df.iloc[i]['Exploratory'] == Exploratory) and (df.iloc[i]['Foreign languages'] == ForeignLanguage) and (df.iloc[i]['Fav genre'] == FavoriteGenre ) and (df.iloc[i]['Instrumentalist'] == Instrumentalist ) and (df.iloc[i]['Composer'] == Composer ) and (df.iloc[i]['Primary streaming service'] == PrimaryStreamingService ) and (df.iloc[i]['While working'] == WhileWorking ) and (df.iloc[i]['Music effects'] == MusicEffect )):

        if ( found_value == 0):
            counter += 1
            found_value = 1

        genre_1 = df.iloc[i]['Frequency [Classical]']
        genre_2 = df.iloc[i]['Frequency [Country]']
        genre_3 = df.iloc[i]['Frequency [EDM]']
        genre_4 = df.iloc[i]['Frequency [Gospel]']
        genre_5 = df.iloc[i]['Frequency [Hip hop]']
        genre_6 = df.iloc[i]['Frequency [Jazz]']
        genre_7 = df.iloc[i]['Frequency [K pop]']
        genre_8 = df.iloc[i]['Frequency [Folk]']
        genre_9 = df.iloc[i]['Frequency [Latin]']
        genre_10 = df.iloc[i]['Frequency [Lofi]']
        genre_11 = df.iloc[i]['Frequency [Metal]']
        genre_12 = df.iloc[i]['Frequency [Pop]']
        genre_13 = df.iloc[i]['Frequency [Rap]']
        genre_14 = df.iloc[i]['Frequency [Rock]']
        genre_15 = df.iloc[i]['Frequency [R&B]']
        genre_16 = df.iloc[i]['Frequency [Video game music]']
        duration = df.iloc[i]['Hours per day']
        bpm = df.iloc[i]['BPM']
        
    data_accuracy = 75.0 # Zulässige Abweichung in Prozent

    if (((df.iloc[i]['Hours per day']/df['Hours per day'].max())-float(Duration)) < 0.5 and  (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Anxiety']-float(Anxiety))/float(Anxiety+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Depression']-float(Depression))/float(Depression+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Insomnia'] -float(Insomnia))/float(Insomnia+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['OCD']-float(OCD))/float(OCD+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Music effects'] == MusicEffect )): # and (df.iloc[i]['Exploratory'] == Exploratory) and (df.iloc[i]['Foreign languages'] == ForeignLanguage) and (df.iloc[i]['Fav genre'] == FavoriteGenre ) and (df.iloc[i]['Instrumentalist'] == Instrumentalist ) and (df.iloc[i]['Composer'] == Composer ) and (df.iloc[i]['Primary streaming service'] == PrimaryStreamingService ) and (df.iloc[i]['While working'] == WhileWorking ) and (df.iloc[i]['Music effects'] == MusicEffect )):

        if ( found_value == 0):
            counter += 1
            found_value = 1

        genre_1 = df.iloc[i]['Frequency [Classical]']
        genre_2 = df.iloc[i]['Frequency [Country]']
        genre_3 = df.iloc[i]['Frequency [EDM]']
        genre_4 = df.iloc[i]['Frequency [Gospel]']
        genre_5 = df.iloc[i]['Frequency [Hip hop]']
        genre_6 = df.iloc[i]['Frequency [Jazz]']
        genre_7 = df.iloc[i]['Frequency [K pop]']
        genre_8 = df.iloc[i]['Frequency [Folk]']
        genre_9 = df.iloc[i]['Frequency [Latin]']
        genre_10 = df.iloc[i]['Frequency [Lofi]']
        genre_11 = df.iloc[i]['Frequency [Metal]']
        genre_12 = df.iloc[i]['Frequency [Pop]']
        genre_13 = df.iloc[i]['Frequency [Rap]']
        genre_14 = df.iloc[i]['Frequency [Rock]']
        genre_15 = df.iloc[i]['Frequency [R&B]']
        genre_16 = df.iloc[i]['Frequency [Video game music]']
        duration = df.iloc[i]['Hours per day']
        bpm = df.iloc[i]['BPM']

    data_accuracy = maximal_accuracy # Zulässige Abweichung in Prozent)
    if (((df.iloc[i]['Hours per day']/df['Hours per day'].max())-float(Duration)) < 0.5 and  (df.iloc[i]['Age']-float(Age))/float(Age)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Anxiety']-float(Anxiety))/float(Anxiety+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Depression']-float(Depression))/float(Depression+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Insomnia'] -float(Insomnia))/float(Insomnia+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['OCD']-float(OCD))/float(OCD+1)*100.0 < (100.0 - data_accuracy) and (df.iloc[i]['Music effects'] == MusicEffect )): #and (df.iloc[i]['Exploratory'] == Exploratory) and (df.iloc[i]['Foreign languages'] == ForeignLanguage) and (df.iloc[i]['Fav genre'] == FavoriteGenre ) and (df.iloc[i]['Instrumentalist'] == Instrumentalist ) and (df.iloc[i]['Composer'] == Composer ) and (df.iloc[i]['Primary streaming service'] == PrimaryStreamingService ) and (df.iloc[i]['While working'] == WhileWorking ) and (df.iloc[i]['Music effects'] == MusicEffect )):

        if ( found_value == 0):
            counter += 1
            found_value = 1

        genre_1 = df.iloc[i]['Frequency [Classical]']
        genre_2 = df.iloc[i]['Frequency [Country]']
        genre_3 = df.iloc[i]['Frequency [EDM]']
        genre_4 = df.iloc[i]['Frequency [Gospel]']
        genre_5 = df.iloc[i]['Frequency [Hip hop]']
        genre_6 = df.iloc[i]['Frequency [Jazz]']
        genre_7 = df.iloc[i]['Frequency [K pop]']
        genre_8 = df.iloc[i]['Frequency [Folk]']
        genre_9 = df.iloc[i]['Frequency [Latin]']
        genre_10 = df.iloc[i]['Frequency [Lofi]']
        genre_11 = df.iloc[i]['Frequency [Metal]']
        genre_12 = df.iloc[i]['Frequency [Pop]']
        genre_13 = df.iloc[i]['Frequency [Rap]']
        genre_14 = df.iloc[i]['Frequency [Rock]']
        genre_15 = df.iloc[i]['Frequency [R&B]']
        genre_16 = df.iloc[i]['Frequency [Video game music]']
        duration = df.iloc[i]['Hours per day']
        bpm = df.iloc[i]['BPM']
       
       
genres = [genre_1, genre_2, genre_3, genre_4, genre_5, genre_6, genre_7, genre_8, genre_9, genre_10, genre_11, genre_12, genre_13, genre_14, genre_15, genre_16] 
numGenre = genres.count(StatsClass)
print('genres:' , genres)

print('Number of genres: ' , numGenre)
print('Genres : ', genre_1, genre_2, genre_3, genre_4, genre_5, genre_6, genre_7, genre_8, genre_9, genre_10, genre_11, genre_12, genre_13, genre_14, genre_15, genre_16)
print ('Duration : ' , duration)
print ('BPM: ', bpm)

print('Counter: ' , counter)

# TO DOs: Modellierung der Daten als Funktion der maximalen Genauigkeit (maximal_accuracy)