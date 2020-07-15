---
layout:     post           # 使用的布局（不需要改）
title:      Linux – git command not found 错误解决
subtitle:   git command not found  #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
 
---

## Linux – git: command not found 错误解决

## 出错原因

服务器没有安装GIT，所以导致出错。



## 解决方法

**Centos下使用：**

```
yum install git -y
```

或者

```
yum install -y git
```

两个代码都是一样的，随意的使用一个即可。

 

**Ubuntu/Debian下使用**

```
apt-get install [git](https://www.bxl.me/b/git/) -y
```

即可解决问题。