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

df = pd.read_csv("UserResearch\MusicExperienceData.csv")

print(df.head())

df_reduced = df['Age'].copy()

print(df_reduced)

print("Age statistics:")
print("Count: ", df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: " , df_reduced.std())


print("")
df_reduced = df['Hours per day'].copy()


print("Hours per day statistics:")
print("Count Duration: " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())

print("")
df_reduced = df['BPM'].copy()


print("BPM statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())

print("")
df_reduced = df[df['OCD'] != 0].copy()
df_reduced = df_reduced['OCD']


print("OCD statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")

df_reduced = df[df['Anxiety'] != 0].copy()
df_reduced = df_reduced['Anxiety']


print("Anxiety statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")

df_reduced = df[df['Depression'] != 0].copy()
df_reduced = df_reduced['Depression']


print("Depression statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")
df_reduced = df[df['Insomnia'] != 0].copy()
df_reduced = df_reduced['Insomnia']


print("Insomnia statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())


df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] == 0) & (df['Depression'] != 0) & (df['Insomnia'] == 0)].copy()

#print("Number of healthy cases: ", df_reduced.count())
print(df_reduced)