---
layout:     post           # 使用的布局（不需要改）
title:      在ubuntu更新时，出现错误E Some index files failed to download, they have been ignored, or old ones used inst
subtitle:   在ubuntu更新时，出现错误E Some index files failed to download, they have been ignored, or old ones used inst  #副标题
date:       2020-07-31             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
 
---

# 在ubuntu更新时，出现错误E: Some index files failed to download, they have been ignored, or old ones used inst

在ubuntu更新时，出现错误E: Some index files failed to download, they have been ignored, or old ones used inst

http://www.ilovn.com/topic/ubuntu-update-error-esome-index-files-failed-to-download-they-have-been-ignored-or-old-ones-used-inst/
在ubuntu更新时

执行命令

sudo apt-get update

出现错误E: Some index files failed to download, they have been ignored, or old ones used instead
可以将目录下/var/lib/apt/lists/partial/所有的文件清掉

$ sudo rm /var/lib/apt/lists/* -vf

再次运行apt-get update

[Link](https://blog.csdn.net/tian_ciomp/article/details/51339635)