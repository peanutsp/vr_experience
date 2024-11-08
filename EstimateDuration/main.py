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
    with open("EstimateDuration\GAN-DataModeler.py") as file:

        code = file.read()
        print("Modelling data with GAN-Datamodeler")
        exec(code)

    with open("EstimateDuration\GANEstimator.py") as file:
        code = file.read()
        print("Running estimation with GANEstimator")
        exec(code)
            

        
    



