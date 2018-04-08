import keras
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import random as rnd

data = datasets.load_breast_cancer()
Y = []
for y in data.target:
    tmp = np.zeros(2)
    tmp[y] = 1
    Y.append(tmp)

x_train, x_test, y_train, y_test = train_test_split(data.data, Y, test_size=0.1)

K = 20
indices = rnd.sample(range(len(x_train)), K)

model = Sequential()
f_layer = FuzzyLayer(K, initializer_centers=lambda x: np.transpose(np.array([x_train[i] for i in indices])), input_dim=30)
model.add(f_layer)
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs=3000,
          verbose=1,
          batch_size=100)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)
weights = f_layer.get_weights()
#print(weights)

plt.ion()
plt.show()
plt.clf()
plt.title('load_breast_cancer')
plt.ylabel('x[0]')
plt.xlabel('x[1]')
plt.scatter([a[0] for a in x_train], [a[1] for a in x_train], c=(0,0,0), alpha=0.5,s=1)
plt.scatter(weights[0][0], weights[0][1], c=(1,0,0), alpha=0.8,s=15)
plt.show()
plt.pause(120)

