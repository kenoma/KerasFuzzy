import keras
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.layers import LSTM
from keras.layers import Embedding

# Generate dummy data
import numpy as np

x = []
y = []
x_test = []
y_test = []
x_n = 0.01
l = 3.8

for i in range(0,10000):
    x_old = x_n
    x_n = l * x_n * (1 - x_n)
    x_nplus = l * x_n * (1 - x_n)

    x.append([[x_old], [x_n]])
    y.append([x_nplus])

for i in range(0, 100):
    x_old = x_n
    x_n = l * x_n * (1 - x_n)
    x_nplus = l * x_n * (1 - x_n)

    x_test.append([[x_old], [x_n]])
    y_test.append([x_nplus])
    

x_train = np.array(x)
y_train = np.array(y)

model = Sequential()
model.add(FuzzyLayer(40, input_shape=(2, 1)))
model.add(LSTM(20))
model.add(DefuzzyLayer(1))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae'])

model.fit(x_train, y_train,
          epochs=1000,
          verbose=1,
          batch_size=1)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)

