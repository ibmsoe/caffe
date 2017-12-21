
# About Large Model Support (LMS)

The key idea is to use GPU memory as an application-level cache to host memory so that a large network (e.g., HD-images) can be trained with Caffe as well. 

## Overview
LMS mainly uses the memory FSM in syncedmem.cpp. Basically, all blobs start on host memory, and copied to GPU memory when only needed. After the GPU memory is used, depending on whether it has been modified or it has future usage, it can be copied back to CPU memory or simply discarded. To efficiently utilize GPU memory, LMS implementation keeps a large memory pool so that different blobs from CPU memory can share the same GPU memory chunk. Therefore, at any moment, GPU only holds blobs necessary to process one layer. If the memory requirement from any layer is larger than the GPU memory, then caffe even with LMS will fail as well. Therefore, in theory, LMS should be able to handle an infinite number of layers as long as all the blobs from the largest layer can fit into GPU memory.


## Parameters
LMS has two commandline parameters.

-lms N (in KB): You can enable the large model support with this. If the value set is zero or negative, LMS is off. For example -lms 1000, then any memory chunk larger than 1000KB will be kept in CPU memory, and fetched to GPU memory only when needed for computation. Thus, if you pass a very large value like -lms 10000000000, it will effectively disable the feature while a small value means more aggressive LMS. The value is to control the performance trade-off.
 
-lms_frac <0~1.0>: As a secondary option, there is -lms_frac. This is to delay LMS kick-in until the given percentage of GPU memory is occupied. For example -lms_frac 0.4, then lms doesn't kick in until it sees more than at least 40% of GPU memory is expected to be taken. This is useful when to activate lms partially for a small network.


## Performance Impact
Since processing a layer needs to wait for all the needed blobs to be fetched to GPU, LMS incurs performance penalty. The penalty can be minimized by carefully tuning the parameters. For example, the larger -lms -lms_frac values are, faster caffe with LMS will run but cannot handle a very large design. It is possible to do smart prefetching to minimize the wait time, but the current implementation does not have it, as it effectively decreases the maximum LMS capability (i.e., GPU memory should hold blobs from more than one layer).

Considering that the link speed between CPU and GPU will improve (e.g., NVLink2), the performance penalty from LMS is expected to desecrate over the time.

## Parameter Tuning Guideline
Start with -lms 1000 -lms_frac 0.0. If this causes OOM, it means the minimum working memory need is larger than the GPU memory, thus much can be done. Otherwise, try to increase -lms first as much as you can, then -lms_frac gradually, utill you cannot change either and have maxed out GPU memory (15.5GB). For example, -lms 2000 -lms_frac 0.0, -lms 100000 -lms_frac 0.1, and so on.


# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
