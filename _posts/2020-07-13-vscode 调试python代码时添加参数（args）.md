---
layout:     post           # 使用的布局（不需要改）
title:      vscode 调试python代码时添加参数（args）
subtitle:   vscode加参数调试 #副标题
date:       2020-07-13            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - python
    - 技术


---

# [vscode 调试python代码时添加参数（args）](https://blog.csdn.net/zk0272/article/details/83105574)

1.  打开`Debug->Open Configurations`
    ![image-20200713182958776](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200713182958.png)
2.  在对应的代码块中添加args，如下图（注意参数之间需要用字符串分割开，用空格是不行的）
    ![image-20200713183029569](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200713183029.png)
3.  再次运行，可以看到结果如下图，自定义的命令已经添加进去了
    ![在这里插入图片描述](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200713182916.png)