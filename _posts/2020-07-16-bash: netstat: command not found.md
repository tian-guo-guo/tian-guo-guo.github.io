---
layout:     post           # 使用的布局（不需要改）
title:      bash netstat command not found
subtitle:   bash netstat command not found  #副标题
date:       2020-07-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
    - 数据库
 
---

# bash: netstat: command not found

If you are looking for the `netstat` command and getting error:

```
bash: netstat: command not found
```

This simply means that the relevant package `net-tools` which includes netstat executable is not installed, thus missing. The package `net-tools` may not be installed on your system by default so you need to install it manually.

The package also includes aditional utilisties such as `arp`, `ifconfig`, `netstat`, `rarp`, `nameif` and `route`.
To make `netstat` available on your system simply install the `net-tools` package using the bellow command:

```
# apt-get install net-tools
```

[Link](https://linuxconfig.org/bash-netstat-command-not-found-debian-ubuntu-linux)