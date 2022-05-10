#%%
import keras
import sys
sys.path.insert(0, '../layers')
from fuzzy_layer import FuzzyLayer
from defuzzy_layer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Generate dummy data
import numpy as np

x_train = np.random.normal(5, 15, size=(100, 2))
y_train = np.array( [([1,0,0,0] if a[0]<10 and a[1]<10 else
            [0,1,0,0] if a[0]<10 and a[1]>10 else
            [0,0,1,0] if a[0]>10 and a[1]<10 else [0,0,0,1]) for a in x_train])

model = Sequential()
model.add(FuzzyLayer(16, input_dim=2,
                     initial_centers=[[15,0,15,0,1,1,1,1, 15,0,15,0,1,1,1,1],
                                                   [0,15,15,0,1,1,1,1, 15,0,15,0,1,1,1,1]], 
                     initial_sigmas=[[1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1],
                                                  [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1]]))
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