from FuzzyLayer import FuzzyLayer
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np

x_train = np.random.normal(0, 1, size=(100, 2))
y_train = np.array( [([1,0,0,0] if a[0]<0.5 and a[1]<0.5 else
            [0,1,0,0] if a[0]<0.5 and a[1]>0.5 else
            [0,0,1,0] if a[0]>0.5 and a[1]<0.5 else [0,0,0,1]) for a in x_train])

model = Sequential()
model.add(FuzzyLayer(20, input_dim=2))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10000,
          verbose=1,
          batch_size=100)

print(model.predict( np.array([[0, 0]])))
print(model.predict( np.array([[0, 1]])))
print(model.predict( np.array([[1, 0]])))
print(model.predict( np.array([[1, 1]])))



print('Done')