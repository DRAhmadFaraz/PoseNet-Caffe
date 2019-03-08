# Proper Installation Guide for this Pose_Net

Requirements: 
Ubuntu > 16.04 
Cuda = 10.1 
CUdnn 7.5 For Cuda = 10.1

# For Posenet Caffe Installation:

Steps:
Install Dependencies



**_sudo apt-get update sudo apt-get upgrade_**

**_sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler_**

**_sudo apt-get install --no-install-recommends libboost-all-dev sudo apt-get install libatlas-base-dev_**

**_sudo apt-get install libopenblas-dev sudo apt-get install the python-dev_**

**_sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev sudo apt install python-pip_**

**_pip install --upgrade pip_**

**_mkdir .local/install cd .local/install_**

**_git clone https://github.com/BVLC/caffe.git_**

**_cd caffe-posenet_**

**_cd python_**

**_for req in $(cat requirements.txt); do sudo -H pip install $req; done Copy the Makefile.config or make it_** 

**_cp Makefile.config.example Makefile.config_** 


### If want to make from scratch Makefile.config file Or just copy makefile.config from this repository
gedit Makefile.config_**

The Makefile.config should contain the following lines, so find them and fill them in.
```
PYTHON_INCLUDE := /usr/include/python2.7
/usr/lib/python2.7/dist-packages/numpy/core/include
```
(for some Ubuntu 16.04 users, the path may be different) 
```
PYTHON_INCLUDE := /usr/include/python2.7
/usr/local/lib/python2.7/dist-packages/numpy/core/include WITH_PYTHON_LAYER := 1
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/hdf5/serial
```

Finish file Makefile.config now test the caffe

**_make all_**
If get error (make: *** [.build_release/tools/upgrade_net_proto_binary.bin] Error 1)

**_make clean_**

**Uncomment if you're using OpenCV 3** 
`OPENCV_VERSION := 3
` 
 

**_make test_**
 ![image](https://user-images.githubusercontent.com/38114191/54049933-c63f1680-41ff-11e9-9ae2-2e1e8df713b5.png)



**_make runtest_**

![image](https://user-images.githubusercontent.com/38114191/54049967-dce56d80-41ff-11e9-806b-2d0f30281650.png)

  
**_make pycaffe_**
 
 ![image](https://user-images.githubusercontent.com/38114191/54049987-e8d12f80-41ff-11e9-998d-91c0913ecde9.png)

**_make pytest_**

![image](https://user-images.githubusercontent.com/38114191/54050000-f38bc480-41ff-11e9-9a15-170cf0064727.png)


### Final test 
**_python _**

**_import sys_**

**_sys.path.append('/home/ahmad/Desktop/caffe-posenet/python') _**

**_import caffe_**

![image](https://user-images.githubusercontent.com/38114191/54050025-030b0d80-4200-11e9-9beb-aaf7cf7f2e09.png)

All done…




# Getting Started with posenet now

**_pip install lmdb_**
**_pip install opencv-python_**
**_sudo apt-get install python-sklearn_**
**_sudo apt-get install python-tk_**

**_cd /home/ahmad/Desktop/caffe-posenet/posenet/scripts_**
Create an LMDB localisation dataset with

caffe-posenet/posenet/scripts/create_posenet_lmdb_dataset.py

Change lines 1, 11 & 12 to the appropriate directories.
Test PoseNet with (do according to your path)

```
caffe_root = '/home/ahmad/Desktop/caffe-posenet/' # Change to your directory to caffe-posenet

directory = '/home/ahmad/Desktop/caffe-posenet/posenet/dataset/KingsCollege/' dataset = 'dataset_train.txt'


```
Replace ./include/caffe/util/cudnn.hpp with the latest version of cudnn in caffe, the corresponding cudnn.hpp.
All files in ./include/caffe/layers that start with cudnn, such as cudnn_conv_layer.hpp. Replaced with the corresponding file of the same name in the latest version of caffe.
Replace all files starting with cudnn, such as cudnn_lrn_layer.cu, cudnn_pooling_layer.cpp, cudnn_sigmoid_layer.cu, in ./src/caffe/layer with the corresponding file of the same name in the latest version of caffe.

Switch Pycaffe environment # Because I have two caffes, the official ones and the posenets, which are compiled separately. # import caffe needs to specify which one to import. Sudo gedit ~/.bashrc # Change the caffe path or caffe-posenet path Source ~/.bashrc

Data set introduction
Take KingsCollege as an example. The following contains a sequence of 8 scene images. 2, 3, 7 are used as test sets (dataset_test.txt), 1, 4, 5, 6, 8 as training sets (dataset_train.txt).

Create an lmdb data set Modify the create_posenet_lmdb_dataset.py file under caffe-posenet/posenet/scripts. There are three places, which are 1, 11, 12 lines; Write a picture description here Install lmdb; Sudo pip install lmdb

Create a dataset: python posenet/scripts/create_posenet_lmdb_dataset.py. Write a picture description here The created data set is generated in the caffe-posenet directory, the posenet_dataset_lmdb folder in the following figure;

Create a mean file Create a new file called "create_posenet_mean.sh";

```
#!/usr/bin/env sh 
set -e
PATH=./data
DATA=./data
DBTYPE=lmdb
echo "Computing image mean..."
./build/tools/compute_image_mean -backend=$DBTYPE \
  $PATH/posenet_dataset_$DBTYPE $PATH/imagemean.binaryproto
echo "Done."
```

Put the lmdb dataset generated in the previous step into the caffe-posenet/data directory;

Run create_posenet_mean.sh to get the mean.binaryproto file.


### Add executable permissions
**_chmod 777 posenet/scripts/create_posenet_mean.sh
**_./posenet/scripts/create_posenet_mean.sh

Modify network configuration

Modify the path of the source and mean_file of the layer whose name is data, phase is TEST and TRAIN in "train_kingscollege.prototxt":



Direct test
input the command:

**_PYTHONPATH=/home/ahmad/Desktop/caffe-posenet/python:$PYTHONPATH 

**_python ./posenet/scripts/create_posenet_lmdb_dataset.py_**

**_python ./posenet/scripts/test_posenet.py --model \ posenet/models/train_posenet.prototxt --weights posenet/models/weights_kingscollege.caffemodel --iter 346_**


Test Results:

![image](https://user-images.githubusercontent.com/38114191/54049863-942db480-41ff-11e9-9887-c950ae055859.png)


# Some common issues (Skip these if u dont face it) 

ls /home/ahmad/.local/lib/python2.7/
find /home/ahmad/.local/lib/python2.7/site-packages -name numpy
/home/ahmad/.local/lib/python2.7/site-packages/numpy/core/include ## if error fatal error: hdf5.h: No such file or directory
find /usr/lib -name hdf5
( you will see	/usr/lib/x86_64-linux-gnu/hdf5 ) ## copy these lines in
gksudo gedit Makefile.config
# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/ LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/ ## ImportError: No module named pydot
find /home/ahmad/.local/lib/ -name "pydot*" python
import pydot
## ImportError: No module named pydot cd .local/install/caffe
gedit Makefile.config
cd .local/install/caffe/python vi requirements.txt
/pydot:
:q
sudo -H pip install pydot make pytest
## make: *** [pytest] Error 1 sudo apt-get install graphviz python
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





