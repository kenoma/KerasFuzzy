import keras
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
Y=[]
for y in iris.target:
    tmp = np.zeros(3)
    tmp[y] = 1
    Y.append(tmp)

x_train, x_test, y_train, y_test = train_test_split(iris.data, Y, test_size=0.1)

model = Sequential()
f_layer = FuzzyLayer(3, input_dim=4)
model.add(f_layer)
#model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['binary_accuracy'])

model.fit(np.array(x_train), 
          np.array(y_train),
          epochs=300,
          verbose=1,
          batch_size=2)

score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True) 
print(score)
weights = f_layer.get_weights()
print(weights)

plt.ion()
plt.show()
plt.clf()
plt.title('Iris')
plt.ylabel('x[0]')
plt.xlabel('x[1]')
plt.scatter([a[0] for a in x_train], [a[1] for a in x_train], c=(0,0,0), alpha=0.5,s=1)
plt.scatter(weights[0][0], weights[0][1], c=(1,0,0), alpha=0.8,s=15)
plt.show()
plt.pause(120)

