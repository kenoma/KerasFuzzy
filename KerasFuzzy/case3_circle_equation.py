import keras
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model

# Generate dummy data
import numpy as np

x = []
y = []
x_test = []
y_test = []

for i in range(-10,10, 2):
    for j in range(-10,10, 2):
        x.append([i / 10.0, j / 10.0])
        y.append((i / 10.0) ** 2 + (j / 10.0) ** 2)

for i in range(-10, 10, 3):
    for j in range(-10, 10, 3):
        x_test.append([i / 10.0, j / 10.0])
        y_test.append((i / 10.0) ** 2 + (j / 10.0) ** 2)

x_train = np.array(x)
y_train = np.array(y)

model = Sequential()
model.add(FuzzyLayer(4, input_dim=2))
model.add(Dense(2, activation='sigmoid'))
model.add(DefuzzyLayer(1))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae'])

model.fit(x_train, y_train,
          epochs=10000,
          verbose=1,
          batch_size=100)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)

#import os
#os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(model, to_file='model.png')