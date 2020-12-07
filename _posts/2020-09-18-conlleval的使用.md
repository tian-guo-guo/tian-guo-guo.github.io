---
layout:     post           # 使用的布局（不需要改）
title:      conlleval的使用           # 标题 
subtitle:   conlleval的使用 #副标题
date:       2020-09-18             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-coffee.jpeg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 命名实体识别

---

# conlleval的使用

在命名体识别任务（[NER，Named Entity Recognizer](https://www.clips.uantwerpen.be/conll2003/ner/)）中,Evaluate使用[Perl](https://www.perl.org/)[的conlleval.pl](http://xn--conlleval-9v0w.pl/)
for example:(例子来源于[link](https://svn.spraakdata.gu.se/repos/richard/pub/ml2015_web/assignment3.html))

```
El          Aa  O       O
consejero   a   O       O
de          a   O       B-MISC
Economía    Aa  B-MISC  I-MISC
Industria   Aa  I-MISC  I-MISC
Comercio    Aa  I-MISC  I-MISC
Manuel      Aa  B-PER   B-PER
Amigo       Aa  I-PER   I-PER
12345678
```

为了使用perl的evaluation工具，我们运行如下命令

```
perl conlleval.perl < output_file_name
或者 conlleval.pl < output_file_name
12
```

便可以得到：

```
processed 10 tokens with 2 phrases; found: 2 phrases; correct: 1.
accuracy:  80.00%; precision:  50.00%; recall:  50.00%; FB1:  50.00
             MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  1
              PER: precision: 100.00%; recall: 100.00%; FB1: 100.00  1
1234
```

为了实现上述过程，我们需要安装perl，下载 [conlleval.pl](http://conlleval.pl/)…如下：

#### 环境搭建

-   平台： win10
-   下载地址： [link](https://www.activestate.com/products/activeperl/downloads/)（ps：perl的window的版本有ActiveState Perl，Strawberry Perl，初学者建议前者。之前又看到原因，等我再找到放个链接吧）
-   安装就是一路狂点next，注意安装的路径，我默认的c盘
    安装完成后会有两个文件
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181229153446813.png)

#### 2.hello world

一般第一个程序都是输出hello world，不过他自己有这个例子，在`C:\Perl64\eg`里有个`example.jl`,用记事本打开就是
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181229155545842.png)
我们打开cmd，切换到`C:\Perl64\eg`这个路径输入

```c
perl example.pl
```

输出：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181229153832212.png)

#### 3.conlleval.perl的使用

原本可以直接下载 [conlleval](https://www.aflat.org/conll2000/chunking/conlleval.txt)。

若官方链接挂了, 可参考此处: [conlleval.pl](https://www.clips.uantwerpen.be/conll2000/chunking/)

![22](https://img-blog.csdnimg.cn/20181229160115506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)

将下载下来的 txt 文档,改名为 [conlleval.pl](http://conlleval.pl/) 或者任何你喜欢的。然后放到`C:\Perl64\eg`里。

自己生成一个测试用的data

```
North B-MISC B-MISC
African E-MISC B-MISC
we O O
Grand B-MISC I-MISC
Prix E-MISC E-MISC
we O O
123456
```

保存为`dataset.txt`依旧放在`C:\Perl64\eg`里。
在cmd里面输入

```c
conlleval.pl < dataset.txt
```

输出
![23](https://img-blog.csdnimg.cn/20181229160542117.png)

#### 4.julia实现的chunk的evaluate

待更新

[Link](https://blog.csdn.net/qq_36097393/article/details/85339553)

