---
layout:     post           # 使用的布局（不需要改）
title:      软性上下文数据增强技术（Soft Contextual Data Augmentation, SCA）           # 标题 
subtitle:   软性上下文数据增强技术（Soft Contextual Data Augmentation, SCA） #副标题
date:       2020-10-05             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - nlp
    - paper
    - 机器翻译
    - 数据增强
---

# 软性上下文数据增强技术（Soft Contextual Data Augmentation, SCA）

## 引言

这篇博客主要是介绍论文[Soft Contextual Data Augmentation for Neural Machine Translation](https://arixv.org/pdf/1905.10523)中采用的一种软性文本数据增强技术。

本文将先介绍几种常用的文本增强技术，然后介绍SCA技术。

## 常见的文本数据增强技术

-   Swap：随机交换相邻位置词（字）的位置，判断两个词是否相邻，可自定义一个大小为k的窗口，位置距离在k以内，则相邻，反之，则不相邻
-   Dropout：以一定概率随机删除词（字）
-   Blank：以一定概率随机将词（字）替换成某占位符，如[MASK]
-   Smooth：以一定概率将词（字）替换成按照频率分布从词典中采样得到的词（字）
-   同义词替换
-   LM_sample：以一定概率将词（字）替换成按照语言模型分布从词典中采样得到的词（字）
-   BackTranslation: 回译，常用于机器翻译中

上述方法在一定程度上都能提高性能。Swap，Dropout，Blank和Smooth比较直观，但是都容易导致句子语义的改变，尤其中文这种博大精深的语言，一字之差有时仿佛天堂地狱。举个栗子，体会一下这种南辕北辙之感：

```
copy原句：     你欠他20万
Swap:     他欠你20万
Dropout：  你欠他20
Blank：   你[MASK]他20万
Smooth：  你欠他20年
```

所以后面有人提出了同义词替换。同义词替换虽然保持了语义上的一致性，但是局限性很大。对于中文来说，同义词往往针对的是动词居多。对于名词来说，同义词起的作用就不大了，而且也不可能将“男人”这个词替换成”女人”。为了解决这个问题，丰富增强的数据，类似同义词这个思路，有人选择用语言模型来选择”同义词”，即上面的LM_sample，也叫上下文增强。众所周知，语言模型能够保持两个词语的语义相似性，似乎能满足我们的需求。也有几篇文章是专门讲这个的，甚至还有人用上了BERT，建议看[Kobayashi](https://arxiv.org/pdf/1805.06201.pdf)这篇。具体做法是，使用语言模型来预测，然后取归一化后概率的top_k作为候选词。为了保证替换后的词不会南辕北辙，例如将good替换成bad，作者采用了一个label_aware语言模型，意思就是在保证类别标签不变的情况下找出候选的替换词，想法很别致。

但是SCA一文的作者指出了LM_sample方法也存在一个缺点：为了产生一个与原始句子语义相似但是有丰富变化的句子，需要进行多次采样。比如说，给定一个句子，有N个词需要被替换，那么候选的可作为增强的句子应该是NTopkNTopk倍，当词典过大时，这种方法无法穷尽所有可能的候选（个人觉得这种说法有点绝对，不过谁的效果好谁有理）。这其实引出了另一个问题，数据增强，增强到什么时候最好呢？最耿直的想法是，增强到指标不再提高为止。这里面其实有很多因素，信息冗余、计算力、模型容量等等，而且在实际中，不做足够多的实验是不知道指标还会不会再提升的，这是个无底洞。

回译是一种较为高级的数据增强方法了，推广起来还有seq2seq增强，本文暂不展开，后面专门讲一讲。

## SCA

下面就来看看SCA是怎么做的。

先用一句话总结一下：随机将词替换成该词的***soft\***版本。

那么，接下来的问题就是如何得到一个词的***soft\***版本呢？

下面给出soft的定义：

一个单词的soft版本是对于任意单词w∈V（V是词典）w∈V（V是词典）,其soft版本是它在整个词典上的概率分布。



P(w)=(p1(w),p2(w),…,p|V|(w)P(w)=(p1(w),p2(w),…,p|V|(w)



其中，pj(w)>=0,Σ|V|j=1pj(w)=1pj(w)>=0,Σj=1|V|pj(w)=1

现在有了概率分布，就可以按照这个概率分布来采样了。到这里思想和LM_sample如出一辙。下面是不同的地方，也是实际怎么用。假设E是词向量矩阵，通常词wjwj的词向量就是EjEj,词向量矩阵的一行。而其soft版本则是：



ew=P(w)E=Σ|V|j=0pj(w)Ejew=P(w)E=Σj=0|V|pj(w)Ej



可以看出，这里的ewew是对所有词向量的加权求和。P(w)P(w)是怎么得到的呢？论文中采用的是一个语言模型。比如在预测单词ww时，用softmax归一化后就能得到每个词的概率，这就组成了P(w)P(w).

论文中特别指出的是，这个语言模型是在训练翻译任务的语料上训练的。

## 实现

SCA架构如图1所示。具体怎么实现呢？

![image-20201005172222564](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201005172222.png)

-   先分别基于source和target平行语料训练两个语言模型，是自回归还是自编码都可以，论文中采用的是Tarnsformer，这样embedding就有了
-   正常进行NMT任务。对于要替换的词，用语言模型先来预测一遍，得到概率分布，和embedding做个计算就能得到soft word embedding了，实现起来改下embedding的加载就可以了。





