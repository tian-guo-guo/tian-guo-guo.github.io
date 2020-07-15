---
layout:     post           # 使用的布局（不需要改）
title:      AttributeError module tensorflow.contrib.estimator has no attribute stop_if_no_decrease_hook
subtitle:   tensorflow.contrib.estimator has no attribute stop_if_no_decrease_hook  #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
 
---

# AttributeError: module 'tensorflow.contrib.estimator' has no attribute 'stop_if_no_decrease_hook'

训练[BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)，运行训练命令时报错：

```
AttributeError: module 'tensorflow.contrib.estimator' has no attribute 'stop_if_no_decrease_hook'
```

查阅资料后知道了

```
tf.estimator.experimental.stop_if_no_decrease_hook  改为这个就好了
```

[Link](https://github.com/macanv/BERT-BiLSTM-CRF-NER/issues/203)

