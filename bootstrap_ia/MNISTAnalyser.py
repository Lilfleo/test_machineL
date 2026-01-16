#Y = LABEL
#X = DATA

from  keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


from mnist import train_images


class MNISTAnalyser:
    def __init__(self):

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()


    #TRAITEMENT

    def moyenne_image(self,i):
        mask_train = self.train_labels == int(i)
        mask_test = self.test_labels == int(i)

        train_image = self.train_images[mask_train]
        test_image = self.test_images[mask_test]

        moyenne_train = train_image.mean(axis=0)
        moyenne_test = test_image.mean(axis=0)

        return moyenne_train, moyenne_test


    def traitement(self):

        matrice_train = self.train_images.reshape(-1, 784)
        matrice_train = matrice_train / 255

        matrice_test = self.test_images.reshape(-1, 784)
        matrice_test = matrice_test / 255

        return matrice_train, matrice_test














