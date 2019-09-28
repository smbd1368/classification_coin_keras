
from time import time
import os
import warnings
import numpy as np
from random import shuffle
from skimage.data import imread
from skimage.transform import resize
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
from keras.callbacks import ReduceLROnPlateau, CSVLogger

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

warnings.filterwarnings('ignore')
image = []
path = '/content/drive/My Drive/Colab Notebooks/train/train/'
path2 = '/content/drive/My Drive/Colab Notebooks/test/'

n = 7
j = 0
for name in os.listdir(path):
    img = imread(os.path.join(path, name))

    img = resize(img, (96, 128))
    for i, s in enumerate(name):
        if s is '_':
            n = int(name[0:i])
    image.append([img, n])

shuffle(image)
labels = []
image1 = []
for i in range(len(image)):
    labels.append(image[i][1])
    image1.append(image[i][0])

label = LabelEncoder().fit_transform(labels)
image1 = np.array(image1)
X_train =image1
y_train =label


######################### TEST image ####################################################vv
labels1 = []
img2 =[]
image2 =[]
image21 =[]
for name in os.listdir(path2):
    img2 = imread(os.path.join(path2, name))
    img2 = resize(img2, (96, 128))
    image2.append([img2, -1])

for i in range(len(image2)):
    labels1.append(image2[i][1])
    image21.append(image2[i][0])

image21 = np.array(image21)
X_test =image21
##############################################################################vv




def CNN_model():
    CNN = Sequential()
    CNN.add(Conv2D(32, (3, 3), input_shape=(96, 128, 3), activation='relu'))
    CNN.add(MaxPooling2D(pool_size=(2, 2)))
    CNN.add(Conv2D(64, (3, 3), activation='relu' ,strides=1 ,padding='same'))
    CNN.add(MaxPooling2D(pool_size=(2, 2)))
    CNN.add(Flatten())
    CNN.add(Dense(32, activation='relu', bias_initializer='glorot_uniform', kernel_regularizer=l2(0.01)))
    CNN.add(Dropout(rate=0.15))
    CNN.add(Dense(64, activation='relu', bias_initializer='glorot_uniform', kernel_regularizer=l2(0.01)))
    CNN.add(Dropout(rate=0.15))
    CNN.add(Dense(5, activation='softmax'))
    opt = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    CNN.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])
    CNN.summary()

    return CNN

model = CNN_model()



lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10,
                       mode='auto', epsilon=1e-1, cooldown=5, min_lr=1e-6)
csv = CSVLogger('validation_log.csv', separator=',', append=True)
fold = KFold(n_splits=5)
fscore = []
accuracy = []
prs = []
roc = []
c = 0


aa =fold.split(X_train, y_train)
for train, valid in aa:
    c += 1
    print(c)
    print('\n CNN iteration  : ', c)
    Xtrain = X_train[train]
    ytrain = y_train[train]
    Xvalid = X_train[valid]
    yvalid = y_train[valid]
    ytrain_label = to_categorical(ytrain, 5)
    yvalid_label = to_categorical(yvalid, 5)
    model.fit(Xtrain, ytrain_label, epochs=10, batch_size=24 ,validation_data=(Xvalid, yvalid_label), callbacks=[lr, csv])
    y = model.predict(Xvalid, batch_size=24)
    prs.append(average_precision_score(y_true=yvalid_label, y_score=y, average='weighted'))
    for i in range(len(y)):
        for k in range(len(y[i])):
            if y[i][k] > 0.65:
                y[i][k] = 1
            else:
                y[i][k] = 0
    fscore.append(f1_score(y_true=yvalid_label, y_pred=y, average='weighted'))
    accuracy.append(accuracy_score(y_true=yvalid_label, y_pred=y))
model.save('cnn_model.h5')
print('Mean Acc  =  ', np.mean(accuracy))
y_train_label = to_categorical(y_train, 5)
model.fit(X_train, y_train_label, epochs=1, batch_size=24, callbacks=[lr])

yp = model.predict(X_test, batch_size=24)
T = time( ) -t4
yp1 =[]
for y in yp:
    yp1.append(np.argmax(y))
print('Output')
print(yp1)

for i in range(len(yp)):
    for k in range(len(yp[i])):
        if yp[i][k] > 0.65:
            yp[i][k] = 1
        else:
            yp[i][k] = 0


