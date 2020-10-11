---
layout:     post           # 使用的布局（不需要改）
title:      jupyter notebook选择conda环境  #标题
subtitle:   jupyter notebook选择conda环境  #副标题
date:       2020-10-02             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - 服务器
 
---

# jupyter notebook选择conda环境

参考 https://stackoverflow.com/questions/37085665/in-which-conda-environment-is-jupyter-executing

需要安装：

```sql
conda install ipykernel
```

使用：

首先激活对应的conda环境

```html
source activate 环境名称
```

将环境写入notebook的kernel中

```html
python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"
```

然后打开notebook

```html
jupyter notebook
```


浏览器打开对应地址，新建python，就会有对应的环境提示了

[Link](https://blog.csdn.net/u011606714/article/details/77741324)

