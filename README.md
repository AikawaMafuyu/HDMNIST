# HDMNIST
Hyperdimensional Computing implementation with MNIST dataset.

## Instructions

### Preparing the dataset
 - The MNIST dataset can be obtained from http://yann.lecun.com/exdb/mnist/ 
 - Put the **extracted** data and label files in one folder.
 - Modify the mnist_path variable to the folder.


### Installing prerequisites.
 - This application runs udner Python 3. Please have the newest python version installed.
 - This application requires the following packages: pickle, pandas, numpy, scikit-learnand python-mnist. Packages can be installed via pip.

### Run the application.
 - Application can be run by directly executing the python script, e.g. "python3 HDMNIST.py"
 - You can adjust different parameters of HDC, including: Dimension, epochs of retraining, training and testing set sizes and quantization bits.
 - Random train/test split is not implemented yet. You need to implement by yourself.
 - This script will generate files of considerable size in the directory, make sure you have proper access and disk space.
