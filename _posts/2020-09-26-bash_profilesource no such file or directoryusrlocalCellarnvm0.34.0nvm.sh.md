---
layout:     post           # 使用的布局（不需要改）
title:      no such file or directory nvm.sh
subtitle:   no such file or directory nvm.sh  #副标题
date:       2020-09-26             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
 
---

# [~/.bash_profile:source:44: no such file or directory: /usr/local/Cellar/nvm/0.34.0/nvm.sh](https://www.cnblogs.com/QuestionsZhang/p/10960124.html)

1.  异常信息

```
/Users/zhangjin/.bash_profile:source:44: no such file or directory: /usr/local/Cellar/nvm/0.34.0/nvm.sh
```

2.  因为自己的/etc/profile有不存在的配置，注释掉即可

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200926165512.png)

