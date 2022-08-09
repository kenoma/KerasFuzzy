#%%
from cmath import exp
from distutils.filelist import translate_pattern
import tensorflow as tf
import numpy as np
from keras import backend as K
from tensorflow import keras
from fuzzy_layer_2 import FuzzyLayer2
from defuzzy_layer import DefuzzyLayer
#%%
output_dim = 2
input_dim = 2
batch = 1
        
layer = FuzzyLayer2(output_dim, initial_centers= [[2, 3], 
                                                  [4, 5]])
layer.build(input_shape=(batch,input_dim))
        
assert layer.R[0,0,0] == 1
assert layer.R[0,0,1] == 0
assert layer.R[0,0,2] == -2
assert layer.R[0,1,0] == 0
assert layer.R[0,1,1] == 1
assert layer.R[0,1,2] == -3

assert layer.R[1,0,0] == 1
assert layer.R[1,0,1] == 0
assert layer.R[1,0,2] == -4
assert layer.R[1,1,0] == 0
assert layer.R[1,1,1] == 1
assert layer.R[1,1,2] == -5

x = tf.convert_to_tensor([[2.0, 3.0]])

y = layer.call(x)
assert y[0][0] == 1
assert y[0][1] == tf.exp(-8.0)


# %%
output_dim = 2
input_dim = 2
batch = 3
        
layer = FuzzyLayer2(output_dim, initial_centers= [[2, 3], 
                                                  [4, 5]])
layer.build(input_shape=(batch,input_dim))
        
assert layer.R[0,0,0] == 1
assert layer.R[0,0,1] == 0
assert layer.R[0,0,2] == -2
assert layer.R[0,1,0] == 0
assert layer.R[0,1,1] == 1
assert layer.R[0,1,2] == -3

assert layer.R[1,0,0] == 1
assert layer.R[1,0,1] == 0
assert layer.R[1,0,2] == -4
assert layer.R[1,1,0] == 0
assert layer.R[1,1,1] == 1
assert layer.R[1,1,2] == -5

x = tf.convert_to_tensor([[ 2.0, 3.0],
                          [ 1.0, 2.0],
                          [ 0.0, 1.0]])

y = layer.call(x)
assert y[0][0] == 1
assert y[0][1] == tf.exp(-8.0)
assert y[1][0] == tf.exp(-2.0)
assert y[1][1] == tf.exp(-18.0)
assert y[2][0] == tf.exp(-8.0)
assert y[2][1] == tf.exp(-32.0)
# %%
output_dim = 3
input_dim = 4
batch = 2
        
layer = FuzzyLayer2(output_dim, initial_centers= [[0, 1, 2, 3], 
                                                  [4, 5, 6, 7], 
                                                  [8, 9, 10, 11]])
layer.build(input_shape=(batch,input_dim))
        
assert layer.R[0,0,0] == 1
assert layer.R[0,0,1] == 0
assert layer.R[0,0,2] == 0
assert layer.R[0,0,3] == 0
assert layer.R[0,0,4] == 0

assert layer.R[0,1,0] == 0
assert layer.R[0,1,1] == 1
assert layer.R[0,1,2] == 0
assert layer.R[0,1,3] == 0
assert layer.R[0,1,4] == -1

assert layer.R[0,2,0] == 0
assert layer.R[0,2,1] == 0
assert layer.R[0,2,2] == 1
assert layer.R[0,2,3] == 0
assert layer.R[0,2,4] == -2

assert layer.R[0,3,0] == 0
assert layer.R[0,3,1] == 0
assert layer.R[0,3,2] == 0
assert layer.R[0,3,3] == 1
assert layer.R[0,3,4] == -3

assert layer.R[1,0,0] == 1
assert layer.R[1,0,1] == 0
assert layer.R[1,0,2] == 0
assert layer.R[1,0,3] == 0
assert layer.R[1,0,4] == -4

assert layer.R[1,1,0] == 0
assert layer.R[1,1,1] == 1
assert layer.R[1,1,2] == 0
assert layer.R[1,1,3] == 0
assert layer.R[1,1,4] == -5

assert layer.R[1,2,0] == 0
assert layer.R[1,2,1] == 0
assert layer.R[1,2,2] == 1
assert layer.R[1,2,3] == 0
assert layer.R[1,2,4] == -6

assert layer.R[1,3,0] == 0
assert layer.R[1,3,1] == 0
assert layer.R[1,3,2] == 0
assert layer.R[1,3,3] == 1
assert layer.R[1,3,4] == -7

x = tf.convert_to_tensor([[ 0.0,  1.0, 2.0,  3.0],
                          [ 8.0,  9.0, 10.0, 11.0]])

y = layer.call(x)

assert y[0][0] == 1
assert y[1][2] == 1
# %%
output_dim = 2
input_dim = 2
batch = None
        
layer = FuzzyLayer2(output_dim, initial_centers= [[2, 3], 
                                                  [4, 5]])
layer.build(input_shape=(batch,input_dim))
        
assert layer.R[0,0,0] == 1
assert layer.R[0,0,1] == 0
assert layer.R[0,0,2] == -2
assert layer.R[0,1,0] == 0
assert layer.R[0,1,1] == 1
assert layer.R[0,1,2] == -3

assert layer.R[1,0,0] == 1
assert layer.R[1,0,1] == 0
assert layer.R[1,0,2] == -4
assert layer.R[1,1,0] == 0
assert layer.R[1,1,1] == 1
assert layer.R[1,1,2] == -5

x = tf.convert_to_tensor([[ 2.0, 3.0],
                          [ 1.0, 2.0],
                          [ 0.0, 1.0]])

y = layer.call(x)
assert y[0][0] == 1
assert y[0][1] == tf.exp(-8.0)
assert y[1][0] == tf.exp(-2.0)
assert y[1][1] == tf.exp(-18.0)
assert y[2][0] == tf.exp(-8.0)
assert y[2][1] == tf.exp(-32.0)
# %%
from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

output_dim = 3
batch = 10
iris = datasets.load_iris()
Y=[]
for y in iris.target:
    tmp = np.zeros(3)
    tmp[y] = 1
    Y.append(tmp)

x_train, x_test, y_train, y_test = train_test_split(iris.data, Y, test_size=0.15)

indices = rnd.sample(range(len(x_train)), output_dim)
model = Sequential()
f_layer = FuzzyLayer2(output_dim, initial_centers=np.array([x_train[i] for i in indices]))
model.add(f_layer)
model.add(DefuzzyLayer(3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs = 4000,
          verbose = 0,
          batch_size=batch)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=1) 
print(score)
weights = f_layer.get_weights()


# %%
#colors for centroids and classes does not match
for pr in [[0,1],[1,2],[2,3]]:
    plt.ion()
    plt.show()
    plt.clf()
    plt.title('Iris')
    plt.ylabel(f'x[{pr[0]}]')
    plt.xlabel(f'x[{pr[1]}]')
    line_color = ["blue", "red", "purple"]
    plt.scatter([a[pr[0]] for a in x_test], [a[pr[1]] for a in x_test], c=[line_color[np.argmax(a)] for a in model.predict(np.array(x_test))], alpha=0.9, s=2)
   
    for odim in range(output_dim):
        origin = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0,0,0,0,1]))
        ort_1 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([ 1,0,0,0,1]))
        ort_2 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([-1,0,0,0,1]))
        ort_3 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0, 1,0,0,1]))
        ort_4 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0,-1,0,0,1]))
        ort_5 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0,0, 1,0,1]))
        ort_6 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0,0,-1,0,1]))
        ort_7 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0,0,0, 1,1]))
        ort_8 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0,0,1])]), np.array([0,0,0,-1,1]))
        plt.plot([-origin[pr[0]], -ort_1[pr[0]]], [-origin[pr[1]], -ort_1[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_2[pr[0]]], [-origin[pr[1]], -ort_2[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_3[pr[0]]], [-origin[pr[1]], -ort_3[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_4[pr[0]]], [-origin[pr[1]], -ort_4[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_5[pr[0]]], [-origin[pr[1]], -ort_5[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_6[pr[0]]], [-origin[pr[1]], -ort_6[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_7[pr[0]]], [-origin[pr[1]], -ort_7[pr[1]]], c =line_color[odim], linewidth=2)
        plt.plot([-origin[pr[0]], -ort_8[pr[0]]], [-origin[pr[1]], -ort_8[pr[1]]], c =line_color[odim], linewidth=2)
