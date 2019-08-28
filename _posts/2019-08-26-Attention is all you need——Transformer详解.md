---
layout:     post                    # 使用的布局（不需要改）
title:      Attention is all you need——Transformer详解              # 标题 
subtitle:   Transformer详解（附pytorch及TensorFlow复现代码） #副标题
date:       2019-08-27              # 时间
author:     甜果果                      # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 学习日记
    - paper
    - nlp
    - 机器翻译
---

## Transformer模型详解

原论文地址：[Attention is all you need](https://arxiv.org/abs/1706.03762)

论文翻译中英文对照：[Attention is all you need中英文对照翻译](https://www.yiyibooks.cn/yiyibooks/Attention_Is_All_You_Need/index.html)

中文笔记参考：[详解Transformer](https://zhuanlan.zhihu.com/p/48508221)
            [Transformer图解](https://fancyerii.github.io/2019/03/09/transformer-illustrated/)

英文笔记参考：[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

PyTorch复现代码：[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html#)

TensorFlow复现代码：[tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

学习笔记：
![Transformer_script_1.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190827-Transformer_script_1.jpg)

![Transformer_script_2.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190827-Transformer_script_2.jpg)

![Transformer_script_3.jpg](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/blog/190827-Transformer_script_3.jpg)
