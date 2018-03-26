import keras
from FuzzyLayers import FuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np

x_train = np.random.normal(5, 15, size=(100, 2))
y_train = np.array( [([1,0,0,0] if a[0]<10 and a[1]<10 else
            [0,1,0,0] if a[0]<10 and a[1]>10 else
            [0,0,1,0] if a[0]>10 and a[1]<10 else [0,0,0,1]) for a in x_train])

model = Sequential()
model.add(FuzzyLayer(8, input_dim=2,
                     initializer_centers=lambda x:[[15,0,15,0,1,1,1,1],
                                                   [0,15,15,0,1,1,1,1]], 
                     initializer_sigmas=lambda x:[[1,1,1,1,1,1,1,1],
                                                  [1,1,1,1,1,1,1,1]]))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae', 'acc'])

model.fit(x_train, y_train,
          epochs=10000,
          verbose=1,
          batch_size=100)

print(model.predict( np.array([[5, 5]])))
print(model.predict( np.array([[5, 15]])))
print(model.predict( np.array([[15, 5]])))
print(model.predict( np.array([[15, 15]])))



print('Done')