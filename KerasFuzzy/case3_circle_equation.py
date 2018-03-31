import keras
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt
import math as m
# Generate dummy data
import numpy as np

x = []
y = []
x_test = []
y_test = []
r = 1
for i in np.linspace(0, 2 * m.pi, num=1000):
    x.append([r * m.cos(i), r * m.sin(i)])
    y.append(i)

for i in np.linspace(0, 2 * m.pi, num=333):
    x_test.append([r * m.cos(i), r * m.sin(i)])
    y_test.append(i)

x_train = np.array(x)
y_train = np.array(y)

f_layer = FuzzyLayer(100, input_dim=2)
model = Sequential()
model.add(f_layer)
model.add(Dense(20, activation='sigmoid'))
model.add(DefuzzyLayer(1))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae'])

model.fit(x_train, y_train,
          epochs=500,
          verbose=1,
          batch_size=100)

y_pred = model.predict(np.array(x_test)) 

plt.ion()
plt.show()
plt.clf()
plt.title('Logistics map')
plt.ylabel('x[n-1]')
plt.xlabel('x[n]')
plt.plot(y_test, c=(0,0,0), alpha=0.5)
plt.plot(y_pred, c=(1,0,0), alpha=0.5)
plt.show()
plt.pause(120)

#weights = f_layer.get_weights()
#print(weights)

#plt.ion()
#plt.show()
#plt.clf()
#plt.title('Logistics map')
#plt.ylabel('x[n-1]')
#plt.xlabel('x[n]')
#plt.scatter([a[0] for a in x_train], [a[1] for a in x_train], c=(0,0,0), alpha=0.5,s=1)
#plt.scatter(weights[0][0], weights[0][1], c=(1,0,0), alpha=0.8,s=15)
#plt.show()
#plt.pause(120)
