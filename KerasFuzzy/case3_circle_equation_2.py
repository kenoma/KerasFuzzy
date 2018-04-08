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
vals = np.linspace(0, 1.9 * m.pi, num=3000)
np.random.shuffle(vals)

for i in vals:
    x.append([r * m.cos(i), r * m.sin(i)])
    y.append(i * m.cos(10 * i))

for i in np.linspace(0, 1.9 * m.pi, num=333):
    x_test.append([r * m.cos(i), r * m.sin(i)])
    y_test.append(i * m.cos(10 * i))

x_train = np.array(x)
y_train = np.array(y)

f_layer = FuzzyLayer(10, input_dim=2)
model = Sequential()
model.add(f_layer)
model.add(Dense(10, activation='sigmoid'))
model.add(DefuzzyLayer(1))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae'])

bweights = f_layer.get_weights()

model.fit(x_train, y_train,
          epochs=10000,
          verbose=1,
          batch_size=100)

y_pred = model.predict(np.array(x_test)) 

weights = f_layer.get_weights()
print(weights)

#plt.ion()
#plt.show()
#plt.clf()
#plt.title('circle')
#plt.ylabel('V')
#plt.xlabel('rad')
#plt.plot(y_test, c=(0,0,0), alpha=0.5)
#plt.plot(y_pred, c=(1,0,0), alpha=0.5)
#plt.show()
#plt.pause(120)

plt.ion()
plt.show()
plt.clf()
plt.title('circle')
plt.ylabel('x')
plt.xlabel('y')
plt.scatter([a[0] for a in x_train], [a[1] for a in x_train], c=(0,0,0), alpha=0.5,s=1)
plt.scatter(weights[0][0], weights[0][1], c=(1,0,0), alpha=0.8,s=15)
plt.scatter(bweights[0][0], bweights[0][1], c=(0,0,1), alpha=0.8, s=10)
plt.show()
plt.pause(100)