import numpy as np
import pickle
import pandas as pd
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances

QBIT = -1

# Loading MNIST dataset and prepare train and test data.
def load_dataset(mnist_path):
    print('Loading MNIST dataset from: ' + mnist_path)
    mndata = MNIST(mnist_path)
    X_train, Y_train = map(np.array, mndata.load_training())
    X_test, Y_test = map(np.array, mndata.load_testing())
    print('Loading Complete!')
    return X_train, Y_train, X_test, Y_test


# Build a look-up table based on given parameters.
def lookup_generate(dim, datatype, n_keys):
    if datatype != 'bipolar':
        raise ValueError('Sorry, currently only supporting bipolar datatype.')
    table = np.random.randint(2, size=(n_keys, dim), dtype=np.int8)
    table[table == 0] = -1
    return table

# Encoding the image into representative hypervecotr
def encode(img, position_table, grayscale_table, dim):
    img_hv = np.zeros(dim, dtype=np.int16)
    for pixel in range(len(img)):
        hv = np.multiply(position_table[pixel], grayscale_table[img[pixel]])
        img_hv = np.add(img_hv, hv)

    return img_hv

# Train the AM
def train(am, X_train, Y_train, position_table, grayscale_table, dim):
    am_ = am.copy()
    # idx = 0
    for img_x, img_y in zip(X_train, Y_train):
        am_[img_y] = np.add(am_[img_y], encode(img_x, position_table, grayscale_table, dim))
        # if idx % 100 == 0:
        #     print(idx)
        # idx = idx + 1
    return am_

# Predict one image with the qhv
def predict_(am, img, position_table, grayscale_table, dim):
    qhv = encode(img, position_table, grayscale_table, dim)
    pred = 0
    d_cos = cosine_distances([qhv, am[pred]])[0][1]

    idx = 1
    for entry in am[1:]:
        d_cos_ = cosine_distances([qhv, entry])[0][1]
        if  d_cos_ < d_cos:
            pred = idx
            d_cos = d_cos_  
        idx = idx + 1

    return pred, qhv

# Predict one image
def predict(am, img, position_table, grayscale_table, dim):
    pred, _ = predict_(am, img, position_table, grayscale_table, dim)
    return pred

# Predict entire test set
def test(am, X_test, Y_test, position_table, grayscale_table, dim):
    Y_pred = []
    #idx = 0
    for img in X_test:
        Y_pred.append(predict(am, img, position_table, grayscale_table, dim))
        # idx = idx + 1
        # if idx % 100 == 0:
        #     print(idx)
    acc = accuracy_score(Y_test, Y_pred)
    print('Testing accuracy is: ' + str(acc))
    return acc

# Retrain the AM
def retrain(am, X_retrain, Y_retrain, position_table, grayscale_table, dim):
    am_ = am.copy()
    for idx in range(len(X_retrain)):
        pred, qhv = predict_(am, X_retrain[idx], position_table, grayscale_table, dim)
        if pred != Y_retrain[idx]:
            am_[pred] = np.subtract(am_[pred], qhv)
            am_[Y_retrain[idx]] = np.add(am_[Y_retrain[idx]], qhv)
    return am_

# Save AM into file
def savemodel(am, position_table, grayscale_table, fpath):
    f = open(fpath, 'wb')
    pickle.dump([am, position_table, grayscale_table], f)
    f.close()
    return 0

# Load AM from file
def loadmodel(fpath):
    f = open(fpath, 'rb')
    am, position_table, grayscale_table = pickle.load(f)
    f.close()
    return am, position_table, grayscale_table


# Quantize AM into different bitlength using the naive method.
def quantize(am, before_bw, after_bw):
    am_ = am.copy()
    if before_bw > after_bw:
        am_ = np.divide(am, 2 ** (before_bw - after_bw))
        am_ = np.rint(am_)
        am_ = am_.astype(np.int16)
    return am_

# Initialization
def main(mode):
    global QBIT
    #mode can be: train / test / retrain (to-be-implemented)
    mnist_path = './mnist/'
    dim = [10000, 5000, 2000, 1000] #Dimensions of HV
    maxval = 256 # Grayscale value range
    imgsize = 28 # Size of MNIST image
    n_class = 10 # Number of classes inside MNIST dataset.
    retraining_epoch = 3 # Number of retraining epochs
    train_size = 1500 # Size of training data.
    test_size = 300 # Size of testing data
    datatype = 'bipolar' # HV type inside item HVs, currently only support bipolar.
    q_bit = [16, 12, 8] # Quantization bits

    X_train, Y_train, X_test, Y_test = load_dataset(mnist_path)
    X_train = X_train[0:train_size]
    Y_train = Y_train[0:train_size]
    X_test = X_test[0:test_size]
    Y_test = Y_test[0:test_size]
    # !IMPORTANT! Currently samples are not randomized, please do that manually so far.
    # train_test_split() to-be-implemented

    if mode == 'train':
        for eachdim in dim:
            fpath = './am_' + str(eachdim)
            position_table = lookup_generate(eachdim, datatype, imgsize*imgsize)
            grayscale_table = lookup_generate(eachdim, datatype, maxval)        
            am = np.zeros((n_class, eachdim), dtype = np.int16)
            am = train(am, X_train, Y_train, position_table, grayscale_table, eachdim)
            for epoch in range(retraining_epoch):
                print('Retraining epoch: ' + str(epoch))
                am = retrain(am, X_train[:train_size], Y_train[:train_size], position_table, grayscale_table, eachdim)
            test(am, X_test[:test_size], Y_test[:test_size], position_table, grayscale_table, eachdim)
            savemodel(am, position_table, grayscale_table, fpath)

    elif mode == 'test':
        for eachqbit in q_bit:
            QBIT = eachqbit
            for eachdim in dim:
                fpath = './am_' + str(eachdim)
                am, position_table, grayscale_table = loadmodel(fpath)
                am_ = quantize(am, 16, eachqbit)            
                acc_ = test(am_, X_test, Y_test, position_table, grayscale_table, eachdim)
                print(acc_)
    return 0

if __name__ == '__main__':
    mode = 'train'
    main(mode)

