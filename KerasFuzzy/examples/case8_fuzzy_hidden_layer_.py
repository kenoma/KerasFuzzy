import sys
sys.path.insert(0, '../layers')
import keras
from fuzzy_layer import FuzzyLayer
from defuzzy_layer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np

x_train = np.random.normal(5, 1, size=(100, 2))
y_train = np.random.normal(5, 15, size=(100, 4))

model = Sequential()
model.add(Dense(2, activation='sigmoid'))
model.add(FuzzyLayer(8))
model.add(Dense(8, activation='sigmoid'))
model.add(DefuzzyLayer(4))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae', 'acc'])

model.fit(x_train, y_train,
          epochs=10000,
          verbose=0,
          batch_size=100)

print(model.predict( np.array([[5, 5]])))
print(model.predict( np.array([[5, 15]])))
print(model.predict( np.array([[15, 5]])))
print(model.predict( np.array([[15, 15]])))



print('Done')
#%%