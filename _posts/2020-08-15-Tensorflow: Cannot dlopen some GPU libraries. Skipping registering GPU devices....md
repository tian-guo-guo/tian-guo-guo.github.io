---
layout:     post           # 使用的布局（不需要改）
title:      Tensorflow Cannot dlopen some GPU libraries. Skipping registering GPU devices...
subtitle:   Tensorflow Cannot dlopen some GPU libraries. Skipping registering GPU devices... #副标题
date:       2020-08-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
    - tensorflow
 
---

# Tensorflow: Cannot dlopen some GPU libraries. Skipping registering GPU devices...

可能的问题为：

1，cuda和安装的tensorflow版本不对应

可参考：[Tensorflow与cuda版本对应关系](https://tensorflow.google.cn/install/source)

2， 未成功加载cuda的动态库，可通过代码如下测试

```python
import tensorflow as tf
tf.test.gpu_device_name()
12
```

如果出现如下错误：

```python
2020-05-26 13:41:11.299037: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory
2020-05-26 13:41:11.299176: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcublas.so.10.0'; dlerror: libcublas.so.10.0: cannot open shared object file: No such file or directory
2020-05-26 13:41:11.299257: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcufft.so.10.0'; dlerror: libcufft.so.10.0: cannot open shared object file: No such file or directory
2020-05-26 13:41:11.299336: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcurand.so.10.0'; dlerror: libcurand.so.10.0: cannot open shared object file: No such file or directory
2020-05-26 13:41:11.299413: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcusolver.so.10.0'; dlerror: libcusolver.so.10.0: cannot open shared object file: No such file or directory
2020-05-26 13:41:11.299490: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcusparse.so.10.0'; dlerror: libcusparse.so.10.0: cannot open shared object file: No such file or directory

1234567
```

可能是未配置 LD_LIBRARY_PATH

```powershell
vim ~/.bashrc
1
```

打开 .bashrc 并在尾部添加如下代码（cuda版本改为你自己的，这里是9.2）

```powershell
export LD_LIBRARY_PATH="/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH" 
1
```

然后使其生效

```powershell
source ~/.bashrc
1
```

3，其他可参考

[tensorflow运行使用CPU不使用GPU](https://blog.csdn.net/xinjieyuan/article/details/103752443)

附：cudnn安装方法

```bash
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
cd /usr/local/cuda/lib64
sudo ln -sf libcudnn.so.7.6.3 libcudnn.so.7
sudo ln -sf libcudnn.so.7 libcudnn.so
sudo ldconfig
12345678
```

参考：[Ubuntu:安装cudnn10.1](https://www.jianshu.com/p/46faad964dc3)

