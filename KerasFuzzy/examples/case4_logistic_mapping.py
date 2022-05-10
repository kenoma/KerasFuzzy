import sys
sys.path.insert(0, '../layers')
import keras
from fuzzy_layer import FuzzyLayer
from defuzzy_layer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt

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

    x.append([x_old, x_n])
    y.append([x_nplus])

for i in range(0, 100):
    x_old = x_n
    x_n = l * x_n * (1 - x_n)
    x_nplus = l * x_n * (1 - x_n)

    x_test.append([x_old, x_n])
    y_test.append([x_nplus])
    

x_train = np.array(x)
y_train = np.array(y)

model = Sequential()
f_layer = FuzzyLayer(5, input_dim=2)
model.add(f_layer)
model.add(Dense(16, activation='sigmoid'))
model.add(DefuzzyLayer(1))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae'])

model.fit(x_train, y_train,
          epochs=1000,
          verbose=0,
          batch_size=100)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)
weights = f_layer.get_weights()
print(weights)

plt.ion()
plt.show()
plt.clf()
plt.title('Logistics map')
plt.ylabel('x[n-1]')
plt.xlabel('x[n]')
plt.scatter([a[0] for a in x_train], [a[1] for a in x_train], c=(0,0,0), alpha=0.5,s=1)
plt.scatter(weights[0][0], weights[0][1], c=(1,0,0), alpha=0.8,s=15)
plt.show()
plt.pause(120)
#%%