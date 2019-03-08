# Proper Installation Guide for this Pose_Net 


For Caffe Installation:

Steps:
sudo apt-get update sudo apt-get upgrade
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev sudo apt-get install libatlas-base-dev
sudo apt-get install libopenblas-dev sudo apt-get install the python-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev sudo apt install python-pip
pip install --upgrade pip

cd python
for req in $(cat requirements.txt); do sudo -H pip install $req; done
 Copy the Makefile.config or make it
 from scratct
cd..
python -m site
/home/ahmad/.local/lib/python2.7/site-packages If want to make from scratch Makefile.config file 
cp Makefile.config.example Makefile.config
gedit Makefile.config
(edit it, enable cpu NOT FOR POSENET, AS POSENET needs GPU)
The following line in the configuration file tells the program to use CPU only for the computations.

CPU_ONLY := 1

The Makefile.config should contain the following lines, so find them and fill them in.

PYTHON_INCLUDE := /usr/include/python2.7
/usr/lib/python2.7/dist-packages/numpy/core/include
(for some Ubuntu 16.04 users, the path may be different) PYTHON_INCLUDE := /usr/include/python2.7
/usr/local/lib/python2.7/dist-packages/numpy/core/include WITH_PYTHON_LAYER := 1
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial

LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/hdf5/serial

Finish file Makefile.config now test the caffe"math_functions.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.

make all

If get error (make: *** [.build_release/tools/upgrade_net_proto_binary.bin] Error 1)

make clean

NVCC src/caffe/util/math_functions.cu
"math_functions.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.

Uncomment if you're using OpenCV 3 
OPENCV_VERSION := 3


make test


make runtest


make pycaffe


make pytest


# python final test 
python 
import sys
sys.path.append('/home/ahmad/caffe-posenet/python') 
import caffe


All done…


# Getting Started with posenet now

pip install lmdb
pip install opencv-python
sudo apt-get install python-sklearn
sudo apt-get install python-tk
cd /home/ahmad/Desktop/caffe-posenet/posenet/scripts
Create an LMDB localisation dataset with

caffe-posenet/posenet/scripts/create_posenet_lmdb_dataset.py

Change lines 1, 11 & 12 to the appropriate directories.
Test PoseNet with

caffe_root = '/home/ahmad/Desktop/caffe-posenet/' # Change to your directory to caffe-posenet
directory = '/home/ahmad/Desktop/caffe-posenet/posenet/dataset/KingsCollege/' dataset = 'dataset_train.txt'

using the command

PYTHONPATH=/home/ahmad/caffe-posenet/python:$PYTHONPATH python 
python ./posenet/scripts/create_posenet_lmdb_dataset.py
python ./posenet/scripts/test_posenet.py --model \ posenet/models/train_posenet.prototxt --weights posenet/models/weights_kingscollege.caffemodel --iter 346
Replace ./include/caffe/util/cudnn.hpp with the latest version of cudnn in caffe, the corresponding cudnn.hpp.
All files in ./include/caffe/layers that start with cudnn, such as cudnn_conv_layer.hpp. Replaced with the corresponding file of the same name in the latest version of caffe.
Replace all files starting with cudnn, such as cudnn_lrn_layer.cu, cudnn_pooling_layer.cpp, cudnn_sigmoid_layer.cu, in ./src/caffe/layer with the corresponding file of the same name in the latest version of caffe.



Switch Pycaffe environment # Because I have two caffes, the official ones and the posenets, which are compiled separately. # import caffe needs to specify which one to import. Sudo gedit ~/.bashrc # Change the caffe path or caffe-posenet path Source ~/.bashrc

Data set introduction
Take KingsCollege as an example. The following contains a sequence of 8 scene images. 2, 3, 7 are used as test sets (dataset_test.txt), 1, 4, 5, 6, 8 as training sets (dataset_train.txt).

Create an lmdb data set Modify the create_posenet_lmdb_dataset.py file under caffe-posenet/posenet/scripts. There are three places, which are 1, 11, 12 lines; Write a picture description here Install lmdb; Sudo pip install lmdb


Create a dataset: python posenet/scripts/create_posenet_lmdb_dataset.py Write a picture description here The created data set is generated in the caffe-posenet directory, the posenet_dataset_lmdb folder in the following figure;
 


Create a mean file Create a new file called "create_posenet_mean.sh" in script folder;

#!/usr/bin/env sh 
set -e
PATH=./data
DATA=./data
DBTYPE=lmdb
echo "Computing image mean..."
./build/tools/compute_image_mean -backend=$DBTYPE \
  $PATH/posenet_dataset_$DBTYPE $PATH/imagemean.binaryproto
echo "Done."

Put the lmdb dataset generated in the previous step into the caffe-posenet/data directory;

Run create_posenet_mean.sh to get the im

Add executable permissions
chmod 777 posenet/scripts/create_posenet_mean.sh
./posenet/scripts/create_posenet_mean.sh

Rename the data and the mean file, prefix plus train_;



Modify network configuration

Modify the path of the source and mean_file of the layer whose name is data, phase is TEST and TRAIN in "train_kingscollege.prototxt":

Direct test
input the command:
sudo python posenet/scripts/test_posenet.py --model posenet/models/train_kingscollege.prototxt --weights posenet/models/weights_kingscollege.caffemodel --iter 8

Test Results:



# Some common issues (Skip these if u dont face it) ls /home/ahmad/.local/lib/python2.7/
find /home/ahmad/.local/lib/python2.7/site-packages -name numpy
/home/ahmad/.local/lib/python2.7/site-packages/numpy/core/include ## if error fatal error: hdf5.h: No such file or directory
find /usr/lib -name hdf5
( you will see	/usr/lib/x86_64-linux-gnu/hdf5 ) ## copy these lines in
gksudo gedit Makefile.config
# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/ LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/ ## ImportError: No module named pydot
find /home/ahmad/.local/lib/ -name "pydot*" python
import pydot
# ImportError: No module named pydot cd .local/install/caffe
gedit Makefile.config
cd .local/install/caffe/python vi requirements.txt
/pydot:
:q
sudo -H pip install pydot make pytest
# make: *** [pytest] Error 1 sudo apt-get install graphviz python
import pydot exit()
make pytest
cd .local/install/caffe/python ls caffe
The build process will fail in Ubuntu 16.04. Edit the Makefile with an editor such as kate ./Makefile
or
gksudo gedit Makefile and replace this line:
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) with the following line
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
Also, open the file CMakeLists.txt and add the following line:

# ---[ Includes
set(${CMAKE_CXX_FLAGS} "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")

if you encounter a missing CUDA error with CUDA version 8.0, find this line in the Makefile.config: CUDA_DIR := /usr/local/cuda
Add Matlab path if you want,
# This is required only if you will compile the matlab interface. # MATLAB directory should contain the mex binary in /bin.
MATLAB_DIR := /usr/local/MATLAB/R2016a/ # MATLAB_DIR := /Applications/R2016a.app

















Median error  1.0539548993110657 m  and  3.9976051339367977 degrees.
