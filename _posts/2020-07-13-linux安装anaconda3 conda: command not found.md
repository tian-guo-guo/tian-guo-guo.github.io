---
layout:     post           # 使用的布局（不需要改）
title:      linux安装anaconda3 conda command not found
subtitle:   conda command not found #副标题
date:       2020-07-13             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
 
---

# linux安装anaconda3 conda: command not found

在终端输入conda info --envs检验anaconda是否安装成功，发现报错：conda: command not found

原因是因为~/.bashrc文件没有配置好

```
vim ~/.bashrc
```

在最后一行加上：

```
export PATH=~/anaconda3/bin:$PATH
```

其中，将“~/anaconda3/bin”替换为你实际的安装路径。保存。

刷新环境

```
source ~/.bashrc
```

成功。