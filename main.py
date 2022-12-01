import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn .inspection import DecisionBoundaryDisplay
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
s
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
n_neighbors = 5

#Leer el archivo con los datos
dataset = pd.read_csv('datosColor.csv')

#Caracteristicas
x = dataset[['b', 'g','r']].values

#Clases
y = dataset['clasificacion'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.70, random_state = 42)
# el mapa de color
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ["#FFOOOO", "#OOFFOO", "#OOOOFF"]


#Uniforme todos los vecinos tienen el mismo peso
#Distance: los vecinos que estan mas cerca tienen mayor peso
for weights in["uniform", "distance"]:
    nca = NeighborhoodComponentsAnalysis(random_state = 42)
    knn = KNeighborsClassifier(n_neighbors)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(x_train, y_train)
    resultado = nca_pipe.score(x_test, y_test)
    print(resultado)

    # Aumento el rango para en rojo y naranja
    redBajo1 = np.array([0, 100, 20], np.uint8)
    redAlto1 = np.array([20, 255, 255], np.uint8)
    redBajo2 = np.array([175, 100, 20], np.uint8)
    redAlto2 = np.array([356, 255, 255], np.uint8)

cap = cv.VideoCapture(0)
while True:
  ret,frame=cap.read()
  if ret==True:
    cv.imshow('frame', frame)
    original = frame.copy()
    frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    maskRed1 = cv.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv.add(maskRed1, maskRed2)
    maskRedvis = cv.bitwise_and(frame, frame, mask= maskRed)  
    cv.imshow('maskRedvis', maskRedvis)
    #convierte de RGB a HSV
    rgb = cv.cvtColor(maskRedvis, cv.COLOR_HSV2RGB)
    #convierte la imagen a gris
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    #Elimina pequeños ruidos
    gray = cv.GaussianBlur(gray, (5, 5), 0);
    #conivierte la imagen en binaria
    ret, thresh = cv.threshold(gray, 100, 255, 0)
    
    #muestra la imagen en binario
    cv.imshow('thresh', thresh)
    #busca los contornos
    contornos,_ = cv.findContours(thresh,  cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contornos, -1, (255,0,0), 3)
    cv.imshow('frame',frame)
    for c in contornos:
        area = cv.contourArea(c)
        if area > 3000:
            M = cv.moments(c)
            if (M["m00"]==0): M["m00"]=1
            x = int(M["m10"]/M["m00"])
            y = int(M['m01']/M['m00'])
            start_point = (x-1, y-1)
            end_point = (x+1, y+1)
            color = (0, 0, 255)
            thickness = 2
            height, width, channels = frame.shape
            #selecciona nueve pixeles al rededor del pixel del centro para calcular el rgb
            if((x+1) <= width and (x-1) >= 0 and (y+1) <= height and (y-1) >= 0):
              frame = cv.rectangle(frame, start_point, end_point, color, thickness)
              pixel = original[y,x]
              #print(pixel)
              #print(pixel[0])#B
              #print(pixel[1])#G
              #print(pixel[2])#R
              b = 0
              g = 0
              r = 0
              for j in range(-1,2):
                for i in range(-1,2):
                  pixel = original[y+j,x+i]
                  b = b + pixel[0]
                  g = g + pixel[1]
                  r = r + pixel[2]
              b = b / 9
              g = g / 9
              r = r / 9
              print("b: " + str(int(b)))
              print("g: " + str(int(g)))
              print("r: " + str(int(r)))
              DatoNuevo = [[b, g, r]]
              prediccion = knn.predict(DatoNuevo)
              print("El nuevo dato es de la clasificación: ", prediccion)
            #cv.circle(frame, (x,y), 7, (0,255,0), -1)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, '{},{}'.format(x,y),(x+10,y), font, 0.75,(0,255,0),1,cv.LINE_AA)
            #nuevoContorno = cv.convexHull(c)
            #cv.drawContours(frame, [nuevoContorno], 0, (255,0,0), 3)
            cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('s'):
      break
    cv.imshow('original', original)
cap.release()
cv.destroyAllWindows()