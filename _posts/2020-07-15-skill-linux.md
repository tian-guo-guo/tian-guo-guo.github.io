---
layout:     post           # 使用的布局（不需要改）
title:      skill-linux
subtitle:   skill-linux  #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
 
---

# skill-linux

## 1. [Linux查看文件夹大小](https://blog.csdn.net/ouyang_peng/article/details/10414499)

```
du -h --max-depth=1
```



### 2. 服务器传送命令

>238上的文件传到237上边去
>
>scp -P 5722 -r /root/tian/OpenNMT-py/self_note root@211.82.97.237:/root/tian/OpenNMT-py/self_note
>
>(在238输命令) (-P是5722 -p是22) (237端口号) (238的路径)(237的路径)

scp -P 5722 -r /root/tian/OpenNMT-py/data root@211.82.97.237:/root/tian/OpenNMT-py/data

## 3. Linux统计文本行数

  - c 统计字节数。
  - l 统计行数。
  - w 统计字数。

```
-c 统计字节数。
-l 统计行数。
-m 统计字符数。这个标志不能与 -c 标志一起使用。
-w 统计字数。一个字被定义为由空白、跳格或换行字符分隔的字符串。
-L 打印最长行的长度。
-help 显示帮助信息
–version 显示版本信息
```

eg:

```
$ wc - lcw file1 file2
4 33 file1
7 52 file2
11 11 85 total
```

## 4. 指定GPU卡2

```
CUDA_VISIBLE_DEVICES=0 
```



## 5. Anaconda

创建环境：`conda create -n py36 python=3.6`  //下面是创建python=3.6版本的环境，取名叫py36 

删除环境：`conda remove -n py36 --all`

激活环境：`conda activate py36`   (conda4之前的版本是：source activate py36 ) //下面这个py36是个环境名

退出环境：`conda deactivate`  (conda4之前的版本是：source deactivate )