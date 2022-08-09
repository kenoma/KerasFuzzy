#%%
import sys
sys.path.insert(0, '../layers')
from fuzzy_layer_2 import FuzzyLayer2
from defuzzy_layer_2 import DefuzzyLayer2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import datetime
np.random.seed(2)
random_seed = 2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import rmsprop_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K 
from tensorflow.python.framework.ops import disable_eager_execution

K.clear_session()
disable_eager_execution()
sns.set(style='dark', context='notebook', palette='deep')
#%%
train = pd.read_csv("./digit-recognizer/train.csv")
test = pd.read_csv("./digit-recognizer/test.csv")

Y_train = train["label"]

X_train = train.drop(labels = ["label"], axis = 1) 
del train 
g = sns.countplot(Y_train)
Y_train.value_counts()
#%%
X_train.isnull().any().describe()
#%%
test.isnull().any().describe()
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
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=13,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.05, # Randomly zoom image 
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)
#%%
encoder_inputs = keras.Input(shape = (28, 28, 1))
latent_dim = 2

x = layers.Conv2D(32, 5, padding='same', activation='relu')(encoder_inputs)
x = layers.Conv2D(32, 5, padding='same', activation='relu', strides=(2, 2))(x)
#x = layers.Dropout(0.01)(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu', strides=(2, 2))(x)
#x = layers.Dropout(0.01)(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="sigmoid")(x)
z_mu = layers.Dense(latent_dim, activation="sigmoid")(x)
z_log_sigma = layers.Dense(latent_dim, activation="sigmoid")(x)

def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mu + tf.keras.backend.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mu, z_log_sigma])
decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
decoder_z = keras.Model(decoder_input, x)
decoder = decoder_z(z)

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, decoder):
        x = tf.keras.backend.flatten(x)
        decoder = tf.keras.backend.flatten(decoder)
        xent_loss = keras.metrics.binary_crossentropy(x, decoder)
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        decoder = inputs[1]
        loss = self.vae_loss(x, decoder)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([encoder_inputs, decoder])

vae = Model(encoder_inputs, y)
vae.compile(optimizer='rmsprop', loss=None, metrics=["accuracy"], experimental_run_tf_function=False)
vae.summary()
#%%
epochs = 100
batch_size = 100
# vae.fit(X_train,
#             y = None, 
#             shuffle = True, 
#             verbose=2, 
#             epochs=epochs, 
#             batch_size=batch_size)
log_dir = "d:/projects/KerasFuzzy/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history_vae = vae.fit(datagen.flow(X_train,y=None, batch_size=batch_size),
                              epochs = epochs, 
                              shuffle = True, 
                              verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[tensorboard_callback])

encoder = Model(encoder_inputs, z_mu)
x_valid_noTest_encoded = encoder.predict(X_val, batch_size=batch_size)
plt.figure(figsize=(10, 10))
plt.scatter(x_valid_noTest_encoded[:, 0], x_valid_noTest_encoded[:, 1]) #, c=Y_val, cmap='icefire'
plt.colorbar()
plt.show()
#
#
#
#
#%%
base_model = Model(encoder_inputs, z_mu)
base_model.trainable = False

x = base_model(encoder_inputs, training = False)
x = FuzzyLayer2(500)(x)
#x = DefuzzyLayer2(200)(x)
x = layers.Dense(10, activation="softmax")(x)
model = Model(encoder_inputs, x)
optimizer = rmsprop_v2.RMSprop(learning_rate=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.92, 
                                            min_lr=0.00001)
epochs = 100
batch_size = 86
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,Y_val),
                              verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])
# %%
base_model.trainable = True
optimizer = rmsprop_v2.RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.01, 
                                            min_lr=0.00001)
epochs = 100
batch_size = 86
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,Y_val),
                              verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])

#%%
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'][15:], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'][15:], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_yscale('log')
ax[1].plot(history.history['accuracy'][15:], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'][15:], color='r',label="Validation accuracy")
ax[1].set_yscale('log')
legend = ax[1].legend(loc='best', shadow=True)
# %%
encoder = Model(encoder_inputs, z_mu)
x_valid_noTest_encoded = encoder.predict(X_val, batch_size=batch_size)
plt.figure(figsize=(10, 10))
plt.scatter(x_valid_noTest_encoded[:, 0], x_valid_noTest_encoded[:, 1])#, c=Y_val, cmap='icefire')
plt.colorbar()
plt.show()
# %%
# Look at confusion matrix 

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

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# %%
# Display some error results 
# Errors are difference between predicted labels and true labels
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

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# predict results
results = model.predict(test)

# %%
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen_6.csv", index=False)
# %%

