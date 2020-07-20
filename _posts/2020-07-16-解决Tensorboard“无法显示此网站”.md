---
layout:     post           # 使用的布局（不需要改）
title:      解决Tensorboard“无法显示此网站”
subtitle:   解决Tensorboard“无法显示此网站”  #副标题
date:       2020-07-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
    - tensorflow
 
---

# 解决Tensorboard“无法显示此网站”

# 问题

输入`tensorboard --logdir=./output/ne_WIPO_NER`，服务器返回的地址是`TensorBoard 1.14.0 at http://e1fd7a8d1822:6006/ (Press CTRL+C to quit)`

在浏览器打开显示“无法显示此网站”

# 解决办法

第一步：打开cmd
第二步：输入tensorboard --logdir=path（path为数据所在文件夹路径) --host=127.0.0.1
第三步：复制路径名称，打开chrome，粘贴并搜索

[Link](https://blog.csdn.net/thisiszdy/article/details/84671313)