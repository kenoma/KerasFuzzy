
#               |1
#   (0,1,0,0)   |   (0,0,0,1)
#               |
#----------------------------------
# -1            |                 1
#   (1,0,0,0)   |   (0,0,1,0)
#               |-1

#%%
from ast import Assert
import sys
sys.path.insert(0, '../layers')
from fuzzy_layer import FuzzyLayer
from defuzzy_layer import DefuzzyLayer
from keras.models import Sequential
import numpy as np

x_train = np.random.uniform(-1, 1, size=(1000, 2))
y_train = np.array( [(
            [1,0,0,0] if a[0]<0 and a[1]<0 else
            [0,1,0,0] if a[0]<0 and a[1]>0 else
            [0,0,1,0] if a[0]>0 and a[1]<0 else 
            [0,0,0,1]) for a in x_train])

model = Sequential()

model.add(FuzzyLayer(16, input_dim=2,
                     initial_centers=[
                         [15,0,15,0,1,1,1,1, 15,0,15,0,1,1,1,1],
                         [0,15,15,0,1,1,1,1, 15,0,15,0,1,1,1,1]],
                     initial_sigmas= [
                         [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1]]))
model.add(DefuzzyLayer(4))
#model.add(Dense(4, activation='sigmoid'))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae', 'acc'])

model.fit(x_train, y_train,
          epochs=100,
          verbose=0,
          batch_size=10)

# %%
assert np.argmax(model.predict(np.array([[ 1,  1]]))) == 3
# %%
assert np.argmax(model.predict(np.array([[-1, -1]]))) == 0
# %%
assert np.argmax(model.predict(np.array([[-1,  1]]))) == 1
# %%
assert np.argmax(model.predict(np.array([[ 1, -1]]))) == 2
# %%
