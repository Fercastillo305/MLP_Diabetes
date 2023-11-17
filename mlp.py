#MLP para predicción de diabetes

#importar librerias
import numpy as np
import pandas as pd
#pandas es para manejo de bases de datos y numpy para matrices

import matplotlib.pyplot as plt
#sirve para graficar

from sklearn.model_selection import train_test_split
#para dividirlos en conjunto de prueba y de aprendizaje

from sklearn.preprocessing import StandardScaler
#estandarizar los datos y preprocesarlos

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

#leer los datos
data= pd.read_csv('diabetes.csv')
data.head()

#identiicar la clase de salida, 0 si no tiene y 1 si si

#cuando se utilizan bases de dtos se recomienda utilizr estadisitca descriptiva para una anlisis de los datos
#media, desviacion estandar, etc

#obtener el tamaño de la base de datos
#solo son 8 caracterisitcas y el 9 es por la salida
data.shape

#estadistica descriptiva 
#promedio de edad
data['Age'].mean()

#desviacion estandar
data['Age'].std()

#grafica de frecuencia 
data['Age'].hist()

data.info()
#revisar como los datos solo son numero porque si son letras no corre

data.describe()

data.hist(figsize=(12,8))

este analisis se realiza para hacer una limpieza de datos, por ejemplo, en glucosa y presion hay datos donde marca 0 y eso no es posible




ver que la base de datos sea para clasificacion y que tenga solo numeros

# Limpieza de datos


#Revisar si hay valores nulos =!0

#en esta no hay
print(data.isnull().sum())

analizar si la vriable puede tener 0's o no, por ejemplo en los embarazos 

#eliminar los datos que no pueden ser 0´s y si lo son 
diabetes=data.drop(data[data['Glucose']==0].index)
diabetes.shape

diabetes=diabetes.drop(diabetes[diabetes['BloodPressure']==0].index)
diabetes.shape

diabetes=diabetes.drop(diabetes[diabetes['BMI']==0].index)
diabetes.shape

diabetes=diabetes.drop(diabetes[diabetes['SkinThickness']==0].index)
diabetes.shape

diabetes=diabetes.drop(diabetes[diabetes['Insulin']==0].index)
diabetes.shape

# Aprendizaje automático

x= diabetes.iloc[:,:-1].values
y=diabetes.iloc[:,-1].values
#el - es para tomar todos los datos menos de la ultima columna
#todas las columnas excepto -1 ( con el :)

#ESTANDARIZAR DATOS
#los pone en el mismo rango
x=StandardScaler().fit_transform(x)

#semilla del generador aleatorio
#para tener el mismo datos 
np.random.seed(1234)

#dividir dastos en entrenamiento y prueba
# por lo general se usan 70% para entrenar y 30% para prueba 
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.3)

#definir el perceptron multicapa
#definir los parametros
#solver realiza el algoritmo de optimizacion 
#en este solo hay una capa escondida
#4 perceptrones en la cada escondida

#solver metodo para optimizar
#son 3 neuronas en una capa escondida 

model=MLPClassifier(solver='adam', max_iter=2000,hidden_layer_sizes=3)
#son 4 porque con 4 lineas rectas podemos separar los parametros 
#establecer parametros para el entrenamiento
#revisar el solver

#entrenar el modelo para obtener los pesos
#es lo que hicimos a mano
model.fit(x_train, y_train)

#evaluar el modelo
#esta haciendolo con datos que no ha visto para ver si lo esta haciendo bien
y_pred=model.predict(x_test)
y_pred

score=model.score(x_test,y_test)
score
#cuando se hacen comparaciones de modelos es necesario usar la misma semilla

matrizConfusion=confusion_matrix(y_test, y_pred)
print(matrizConfusion)

#ver si en las clases acerto o non
#se trabaja con prueba 
