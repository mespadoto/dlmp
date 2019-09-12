#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from glob import glob
from keras import applications
from keras import datasets as kdatasets
from skimage import io, transform
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import scipy.io as sio
from scipy.io import arff
import tarfile
import tempfile
import zipfile
import pickle
import pandas as pd
import wget

def load_mat(file):
    data = sio.loadmat(file)

    X = np.rollaxis(data['X'], 3, 0)
    y = data['y'].squeeze()

    return X, y

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def process_cifar10():
    if os.path.exists('data/X_cifar10_img.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.cifar10.load_data()

    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X / 255.0

    X = X[np.isin(y, [1, 5])]
    y = y[np.isin(y, [1, 5])]

    y[y==5] = 0 #dogs
    y[y==1] = 1 #automobiles
    
    model = applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    X_new = model.predict(X, verbose=1, batch_size=512)
    model = None

    np.save('data/X_cifar10_img.npy', X)
    np.save('data/X_cifar10_densenet.npy', X_new)
    np.save('data/y_cifar10.npy', y)

def process_cifar100():
    if os.path.exists('data/X_cifar100.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.cifar100.load_data(label_mode='coarse')

    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X / 255.0
    
    X = X[np.isin(y, [12, 18])]
    y = y[np.isin(y, [12, 18])]

    y[y==12] = 0 #medium-sized mammals
    y[y==18] = 1 #vehicles

    model = applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    X_new = model.predict(X, verbose=1, batch_size=512)
    model = None

    np.save('data/X_cifar100.npy', X_new)
    np.save('data/y_cifar100.npy', y)


def process_mnist():
    if os.path.exists('data/X_mnist.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.mnist.load_data()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X.reshape((-1, 28 * 28)) / 255.0
    y = y.squeeze()

    np.save('data/X_mnist.npy', X)
    np.save('data/y_mnist.npy', y)


def process_fashion():
    if os.path.exists('data/X_fashion.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.fashion_mnist.load_data()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X.reshape((-1, 28 * 28)) / 255.0
    y = y.squeeze()

    np.save('data/X_fashion.npy', X)
    np.save('data/y_fashion.npy', y)

def process_imdb():
    if os.path.exists('data/X_imdb.npy'):
        return

    if not os.path.exists('data/aclImdb_v1.tar.gz'):
        wget.download('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', out='data')

    imdb = tarfile.open('data/aclImdb_v1.tar.gz', 'r:gz')
    tmp_dir = tempfile.TemporaryDirectory()
    imdb.extractall(tmp_dir.name)

    pos_files = glob(os.path.join(
        tmp_dir.name, 'aclImdb/train/pos') + '/*.txt')
    pos_comments = []

    neg_files = glob(os.path.join(
        tmp_dir.name, 'aclImdb/train/neg') + '/*.txt')
    neg_comments = []

    for pf in pos_files:
        with open(pf, 'r') as f:
            pos_comments.append(' '.join(f.readlines()))

    for nf in neg_files:
        with open(nf, 'r') as f:
            neg_comments.append(' '.join(f.readlines()))

    comments = pos_comments + neg_comments
    y = np.zeros((len(comments),)).astype('uint8')
    y[:len(pos_comments)] = 1

    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=500)

    X = tfidf.fit_transform(comments).todense()

    np.save('data/X_imdb.npy', X)
    np.save('data/y_imdb.npy', y)

def process_dogandcats():
    if os.path.exists('data/X_dogsandcats.npy'):
        return
    
    if not os.path.exists('data/kagglecatsanddogs_3367a.zip'):
        wget.download('https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip', out='data')
        f = zipfile.ZipFile('data/kagglecatsanddogs_3367a.zip')
        f.extractall('data')

    side = 128
    model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(side, side, 3), pooling='max')

    dirs = glob('data/PetImages/*')
    
    size = 0
    
    for d in sorted(dirs):
        files = glob(d + '/*.*')
        size += len(files)
        print(d, len(files))
    
    X = np.zeros((size, side, side, 3)).astype('float32')
    y = np.zeros((size,)).astype('int8')
    
    i = 0
    
    for class_id, d in enumerate(sorted(dirs)):
        class_name = os.path.basename(d)
        files = glob(d + '/*.*')
    
        for f in sorted(files):
            try:
                img = io.imread(f)
                img = transform.resize(img, (side, side), preserve_range=True)
                
                if len(img.shape) < 3:
                    print(img.shape)
                    img_new = np.zeros((img.shape[0], img.shape[1], 3)).astype('uint8')
                    
                    for j in range(3):
                        img_new[:,:,j] = np.copy(img)
                    
                    img = img_new
                
                img = (img / 255.0).astype('float32')
    
                print(class_id, class_name, f, i, img.shape)
    
                X[i,:,:,:] = img
                y[i] = class_id
                
            except:
                print('Error processing image %s' % f)
                X[i,:] = 0.0
                y[i] = -1
    
            i += 1

    X = model.predict(X, verbose=1)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    np.save('data/X_dogsandcats.npy', X)
    np.save('data/y_dogsandcats.npy', y)


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        print(str(func))
        globals()[func]()





