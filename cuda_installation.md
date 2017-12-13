## Install tensorflow with gpu library CUDA on Ubuntu 16.04 x64

### System and Software info
1. System: Ubuntu16.04
2. GPU card: Nvidia GeForce GT 620
3. tensoflow-gpu==1.2.1
4. CUDA: 8.0 https://developer.nvidia.com/cuda-downloads 
5. cuDNN: v5.1 https://developer.nvidia.com/rdp/cudnn-download

### References
1. TensofFlow http://www.tensorflow.org
2. Chinese staffs
   [1] http://blog.csdn.net/yichenmoyan/article/details/48679777
   [2] http://blog.csdn.net/niuwei22007/article/details/50439478


### Installation
- Install cuda, and configure path and LD_LIBRARY_PATH
```sh
$ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
$ sudo apt-get update
$ sudo apt-get install cuda
```

- Configure path
``` sh
$ vim ~/.bashrc
$ export PATH=$PATH:/usr/local/cuda-8.0/bin
$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/lib
```

- Install cuDNN
``` sh
$ tar -xvf cudnn-8.0-linux-x64-v5.1.tgz
$ cd cuda
$ sudo cp ./lib64/* /usr/local/cuda/lib64/
$ sudo chmod 755 /usr/local/cuda/lib64/libcudnn*
$ sudo cp ./include/cudnn.h /usr/local/cuda/include/
```

- Install Tensorflow
```sh
$ <sudo> pip3 install <--user> <--update> tenforflow-gpu==1.2.1
```

### Note
Note that cuda 8.0 doesn't support the default g++ version. Install an supported version and make it the default.
``` sh
$ sudo apt-get install g++-4.9

$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10

$ sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
$ sudo update-alternatives --set cc /usr/bin/gcc

$ sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
$ sudo update-alternatives --set c++ /usr/bin/g++
```

