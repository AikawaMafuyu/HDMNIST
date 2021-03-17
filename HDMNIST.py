
import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


mnist_path = './mnist/'
dim = 10000
maxval = 256
imgsize = 28
classes = 10

train_size = 3000
test_size = 2000

# Loading the dataset and split into train and test set.
def load_dataset(mnist_path):
    mndata = MNIST(mnist_path)
    X_train, Y_train = map(np.array, mndata.load_training())
    X_test, Y_test = map(np.array, mndata.load_testing())
    return X_train, Y_train, X_test, Y_test

# Generate item memories (both position and value)
def PositionHV(imgsize, valrange):
    posHV = np.random.randint(2, size=(imgsize * imgsize, dim))
    valHV = np.random.randint(2, size=(valrange, dim))
    posHV[posHV == 0] = -1
    valHV[valHV == 0] = -1
    return posHV, valHV

# Training:
X_train, Y_train, X_test, Y_test = load_dataset(mnist_path)
x_tr, x_te, y_tr, y_te = train_test_split(X_test, Y_test, test_size=0.25, random_state = 9)
posHV, valHV = PositionHV(imgsize, maxval)
RefMemory = np.zeros((classes*2, dim))

imgidx = 0
for img in x_tr:
    imgHV = np.zeros(dim)
    pixidx = 0
    # Encode each img into representative HV
    for pix in img:
        pixHV = np.multiply(posHV[pixidx], valHV[pix])
        imgHV = np.add(imgHV, pixHV)
        pixidx += 1
    # Bipolarize HV
    imgHV[imgHV >= 0] = 1
    imgHV[imgHV < 0] = -1
    # Update AM
    RefMemory[y_tr[imgidx]] = np.add(RefMemory[y_tr[imgidx]], imgHV)
    imgidx += 1
    if imgidx == train_size:
        RefMemory[RefMemory > 0] = 1
        RefMemory[RefMemory <= 0] = -1
        break
        
# Testing
imgidx = 0
y_pred = []
for img in x_te:
    imgHV = np.zeros(dim)
    pixidx = 0
    for pix in img:
        pixHV = np.multiply(posHV[pixidx], valHV[pix])
        imgHV = np.add(imgHV, pixHV)
        pixidx += 1
    imgHV[imgHV >= 0] = 1
    imgHV[imgHV < 0] = -1
    sim = []
    for each_class in RefMemory:
        sim.append(cosine_similarity([each_class, imgHV])[0][1])
        #sim.append(1 - distance.hamming(each_class, imgHV))
    y_pred.append(sim.index(max(sim)))
    imgidx += 1
    if imgidx == test_size:
        break

print(accuracy_score(y_te[0:1000], y_pred))



