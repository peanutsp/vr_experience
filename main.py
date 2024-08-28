################################################################################################################################################
#                                                                                                                                              #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt    #
#                                                                                                                                              #
################################################################################################################################################

# PYTHON ROUTINE zur Steuerung der GAN-Modelle und Berechnung von Immersionslevel

# Initialer Import aller nötigen Libraries
import os
import sys
import math
import pylab
import matplotlib.pyplot as plt
# from GAN-DataModeler import Generator
# Ein Beispiel zur Integration von 'GAN-DataModeler.py'



if __name__ == "__main__":
    
    print("Init")
    with open("GAN-DataModeler.py") as file:

        code = file.read()
        print("Modelling data with GAN-Datamodeler")
        exec(code)

    with open("GANEstimator.py") as file:
        code = file.read()
        print("Running estimation with GANEstimator")
        exec(code)
            
    with open("Correlations.py") as file:
        
        code = file.read()
        print("Calculating correlations")
        exec(code)
        
    with open("VRExperience.py") as file:
        code = file.read()
        print("Calculating accuracies")
        exec(code)
        
    

# Workflow:

# 1. ) Eingabe der Parameter (UserID, Age, Gender, VRHeadset, Duration)
# 2. ) Auslagerung von MotionSickness und Immersionslevel in ein externes Files oder MySQL-Datenbank
# 3. ) Lege eine Schnittstelle (MySQL-Datenbank fest), welche für unterschiedliche Nutzer die berechneten Parameter speichert


# Accuracy mit verschiedenen Data_similarities in GAN-Estimator

