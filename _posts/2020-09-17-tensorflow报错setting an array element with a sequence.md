---
layout:     post           # 使用的布局（不需要改）
title:      tensorflow报错setting an array element with a sequence
subtitle:   tensorflow报错setting an array element with a sequence  #副标题
date:       2020-09-17             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
    - tensorflow
 
---

# tensorflow报错：setting an array element with a sequence

# 问题

在训练bilstm-crf模型的时候，往模型里送word.npy文件时报错setting an array element with a sequence

# 分析

这个问题一般会发生在读取数据的时候，也就是把我们Python里面的数据传递给placeholder的时候会报这个错。

为什么会出现这样的错误，看下面两行代码：

```python
a = tf.constant([[[[1,2,3], [6,7,8], [9,10,1]],
                   [[0,1,2], [2,3,4], [3,4,1]]],

                  [[[1,2,3], [6,7,8], [9,10,1]],
                   [[0,1,2], [2,3,4], [3,4,1]]]], dtype=np.float32)

a = tf.constant([[[[1,2,3,4], [6,7,8], [9,10]], 
                   [[0,1,2,3], [2,3,4], [3,4]]], 

                  [[[1,2,3,4], [6,7,8], [9,10]], 
                   [[0,1,2,3], [2,3,4], [3,4]]]], dtype=np.float32)
```

结果会告诉你第一个语句正确可以执行，第二个语句错误，且爆出下面这样的错误：

```
Traceback (most recent call last):
  File "/home/lc/PycharmProjects/tensorflow/HAN/HAN-text-classification-tf/test.py", line 65, in <module>
    [[0,1,2,3], [2,3,4], [3,4]]]], dtype=np.float32)
  File "/home/lc/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/constant_op.py", line 102, in constant
    tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  File "/home/lc/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/tensor_util.py", line 371, in make_tensor_proto
    nparray = np.array(values, dtype=np_dt)
ValueError: setting an array element with a sequence.
```

这说明什么呢，第二条赋值语句的list元素的shape是不一样的。它不像第一个里面每一个元素长度都是3，反而是2,3,4个元素都有，所以在复制的时候程序并不知道该怎样去做。就将其当做是一个sequence对待，自然是无法赋值的。

其次还有一种情况就是咱们一开始提到的那种，当我们给palceholder传递数据的时候偶尔也会出现这种问题，这时候我们一般会在数据处理的时候将数据使用np.array()进行封装。大概就像这样，使用列表添加append数据的时候就封装一次。

```python
data_x = []

max_sent_in_doc = 30
max_word_in_sent = 30
for perdoc in x_text:
    doc2idx = []
    snt_doc = perdoc.split('。')
    for i,sent in enumerate(snt_doc):
        if i <max_sent_in_doc:
            word2idx =[]
            for j,word in enumerate(sent.split(' ')):
                if j < max_word_in_sent:
                    word2idx.append(np.array(vocab.get(word,UNKNOWN)))
            npad = (0,30-len(word2idx))

            #add padding 
            word2idx = np.pad(word2idx, pad_width=npad, mode='constant', constant_values=0)#padding
            doc2idx.append(np.array(word2idx))
    npad2 =((0,30-len(doc2idx)),(0,0))

    #add padding
    doc2idx = np.pad(doc2idx, pad_width=npad2, mode='constant', constant_values=0)#padding
    data_x.append(np.array(doc2idx))
```

注意data_x.append()和doc2idx.append()两句即可。

[Link](https://blog.csdn.net/liuchonge/article/details/77854689)