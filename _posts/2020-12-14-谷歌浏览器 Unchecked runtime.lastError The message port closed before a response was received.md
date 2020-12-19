---
layout:     post           # 使用的布局（不需要改）
title:     谷歌浏览器 Unchecked runtime.lastError The message port closed before a response was received.
subtitle:   谷歌浏览器 Unchecked runtime.lastError The message port closed before a response was received.  #副标题
date:       2020-12-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - vue
 
---

# 谷歌浏览器 Unchecked runtime.lastError: The message port closed before a response was received.

#### 谷歌浏览器报错

-   版本 `72.0.3626.81（正式版本） （32 位）`
-   报错内容如下
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190201091615587.png)

#### 原因：扩展程序问题

-   建议：打开`chrome://extensions/`，逐一关闭排查
-   以我的为例，发现罪魁祸首是以下扩展程序，最后关闭就好。
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190201091853183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3Mjg1MTkz,size_16,color_FFFFFF,t_70)