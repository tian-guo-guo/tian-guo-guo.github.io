---
layout:     post           # 使用的布局（不需要改）
title:      vscode ssh远程输入密码之后无反应，一直提示输入密码
subtitle:   vscode ssh远程输入密码之后无反应，一直提示输入密码  #副标题
date:       2020-11-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - bugs
    - 服务器
 
---

# vscode ssh远程输入密码之后无反应，一直提示输入密码

解决方法参考[链接](https://github.com/microsoft/vscode-remote-release/issues/2518)
方法：**点击view下的‘command palette’下的’remote-ssh: kill vs code server on host…'**
之后会有一些文件自动下载，不用理会，再次尝试输入密码即可。
![在这里插入图片描述](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201116092607.png)

[Link](https://blog.csdn.net/Mr_Cat123/article/details/107432070)

