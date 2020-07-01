---
layout:     post           # 使用的布局（不需要改）
title:      git pull error Your local changes to the following files would be overwritten by merge
subtitle:   git pull error #副标题
date:       2020-06-24             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
 
---

# git pull “error: Your local changes to the following files would be overwritten by merge”

今天在使用git pull 命令的时候发生了以下报错

[![image.png](http://127.0.0.1:8090/upload/2020/06/image-2cd77f59e0a94691b8bda2beb958ac07.png)](http://127.0.0.1:8090/upload/2020/06/image-2cd77f59e0a94691b8bda2beb958ac07.png)

　　目前git的报错提示已经相关友好了，可以直观的发现，这里可以通过commit的方式解决这个冲突问题，但还是想看看其他大佬是怎么解决这类问题的

　　在网上查了资料和其他大佬的博客，得到了两种解决方法：

方法一、stash

```
1 git stash
2 git commit
3 git stash pop
```

接下来diff一下此文件看看自动合并的情况，并作出相应修改。

git stash: 备份当前的工作区的内容，从最近的一次提交中读取相关内容，让工作区保证和上次提交的内容一致。同时，将当前的工作区内容保存到Git栈中。
git stash pop: 从Git栈中读取最近一次保存的内容，恢复工作区的相关内容。由于可能存在多个Stash的内容，所以用栈来管理，pop会从最近的一个stash中读取内容并恢复。
git stash list: 显示Git栈内的所有备份，可以利用这个列表来决定从那个地方恢复。
git stash clear: 清空Git栈。此时使用gitg等图形化工具会发现，原来stash的哪些节点都消失了。

方法二、放弃本地修改，直接覆盖

```
1 git reset --hard
2 git pull
```

参考原文：https://blog.csdn.net/lincyang/article/details/21519333