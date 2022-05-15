#%%
from cmath import exp
from distutils.filelist import translate_pattern
import tensorflow as tf
import numpy as np
from keras import backend as K
from tensorflow import keras
from fuzzy_layer_2 import FuzzyLayer2
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

output_dim = 7
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
model.add(Dense(3, activation='softmax'))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs = 200,
          verbose = 0,
          batch_size=batch)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=1) 
print(score)
weights = f_layer.get_weights()

# %%
