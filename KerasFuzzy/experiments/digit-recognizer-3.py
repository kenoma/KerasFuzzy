#%%
import sys
sys.path.insert(0, 'D:/projects/KerasFuzzy/KerasFuzzy/layers')
from fuzzy_layer_2 import FuzzyLayer2
from defuzzy_layer_2 import DefuzzyLayer2
from defuzzy_layer import DefuzzyLayer

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
np.random.seed(2)
random_seed = 2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K 
import random
import itertools
#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
train = pd.read_csv("./digit-recognizer/train.csv")
test = pd.read_csv("./digit-recognizer/test.csv")

Y_train = train["label"]

X_train = train.drop(labels = ["label"], axis = 1) 
del train 
#%%
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#%%
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
#%%
g = plt.imshow(X_train[0][:,:,0])

#%%
latent_dim = 3

mnist_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", padding="same")(mnist_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
#x = layers.Dense(32, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
base_model = keras.Model(mnist_inputs, z_mean)
base_model.summary()
#%%


# %%
fuzzy_centroids = 10
# base_model_samples = base_model.predict(X_train)
# init_c = random.sample(list(base_model_samples), fuzzy_centroids)
# init_s = np.empty((fuzzy_centroids, latent_dim))
# init_s.fill(0.1)

x = base_model(mnist_inputs)
x = FuzzyLayer2(fuzzy_centroids, name="fuzzy")(x)
#x = tf.keras.layers.Softmax()(x)
#x = keras.layers.LayerNormalization(axis=1)(x)
x = DefuzzyLayer2(10, name="defuzzy")(x)
x = tf.keras.layers.Softmax()(x)
fuzzy_model = keras.Model(mnist_inputs, x)

optimizer = keras.optimizers.Adam()
fuzzy_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
fuzzy_model.summary()
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.0000001)
epochs = 100
batch_size = 300
datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=13,  
        zoom_range = 0.05, 
        width_shift_range=0.05,  
        height_shift_range=0.05,  
        horizontal_flip=False,  
        vertical_flip=False)  
datagen.fit(X_train)
log_dir = "d:/projects/KerasFuzzy/logs/whole_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = fuzzy_model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,Y_val),
                              verbose = 1, 
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction, tensorboard_callback])
#%%
def plot_label_clusters(vae, data, labels):
    z_means = vae.predict(data)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].scatter(z_means[:, 0], z_means[:, 1], c=labels)
    ax[0].set_xlabel("z[0]")
    ax[0].set_ylabel("z[1]")
    if latent_dim>2:
        ax[1].scatter(z_means[:, 2], z_means[:, 1], c=labels)
        ax[1].set_xlabel("z[2]")
        ax[1].set_ylabel("z[1]")
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(z_means[:, 0], z_means[:, 1], z_means[:, 2], c=labels)
    # ax.set_xlabel('z[0]')
    # ax.set_ylabel('z[1]')
    # ax.set_zlabel('z[2]')
    # ax.view_init(-120, 120)
    
plot_label_clusters(base_model, X_train, [np.argmax(a) for a in Y_train])
#plot_label_clusters(base_model, X_val, [np.argmax(a) for a in Y_val])
#%%
#z_means = base_model.predict(X_val)
#plt.scatter(z_means[:, 0], z_means[:, 1], c=[np.argmax(a) for a in Y_val])
learned_centroids = []
weights = fuzzy_model.get_layer('fuzzy').get_weights()
for odim in range(fuzzy_centroids):
    e = 2
    origin = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,0,0, 1]))
    e1 =  np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([e,0,0, 1]))
    e2 =  np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,e,0, 1]))
    e3 =  np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,0,e, 1]))
    me1 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([-e,0,0, 1]))
    me2 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,-e,0, 1]))
    me3 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,0,-e, 1]))
    plt.plot([-origin[0], -e1[0]],  [-origin[1], -e1[1]],  c = 'black', linewidth=1)
    plt.plot([-origin[0],  -e2[0]], [-origin[1], -e2[1]],  c = 'black',linewidth=1)
    plt.plot([-origin[0],  -e3[0]], [-origin[1], -e3[1]],  c = 'black',linewidth=1)
    plt.plot([-origin[0], -me1[0]], [-origin[1], -me1[1]], c = 'black',linewidth=1)
    plt.plot([-origin[0], -me2[0]], [-origin[1], -me2[1]], c = 'black',linewidth=1)
    plt.plot([-origin[0], -me3[0]], [-origin[1], -me3[1]], c = 'black',linewidth=1)
    learned_centroids.append(origin)

plt.scatter([-a[0] for a in learned_centroids], [-a[1] for a in learned_centroids], c = 'black', alpha=0.9, s=30)

#ax[0].set_xlabel("z[0]")
#ax[0].set_xlabel("z[1]")
    
#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_pred = fuzzy_model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_val,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
results = fuzzy_model.predict(test)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results],axis = 1)
submission.to_csv("cnn_mnist_full_fuzzy.csv", index=False)
# %%
