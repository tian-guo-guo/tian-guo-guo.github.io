---
layout:     post           # 使用的布局（不需要改）
title:      No valid exports main found for @rollup/pluginutils 错误解决
subtitle:   No valid exports main found for @rollup/pluginutils  #副标题
date:       2020-11-04             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - bugs
 
---

# No valid exports main found for @rollup/pluginutils

>   this is because Node.js 13.5.0 does not support this feature of the `"exports"` field. I'd suggest updating Node.js here.

And I can confirm, with node `14.4` I no longer have this problem.

[Link](https://github.com/vitejs/vite/issues/367)



# 怎么更新升级node版本？

**更新升级node版本的方法如下：**

1）首先：查看当前node版本：

```
node –v
```

2）安装n模块：

```
npm install -g n
```

3）升级到指定版本/最新版本（该步骤可能需要花费一些时间）升级之前，可以执行n ls （查看可升级的版本）
如：`n v6.9.1`

或者你也可以告诉管理器，安装最新的稳定版本

```
n stable
```

或者升级到最新版

```
n latest
```

4）安装完成后，查看Node的版本，检查升级是否成功

```
node -v
```

注：如果得到的版本信息不正确，你可能需要重启机器

最后，扩展说明：

有很多同学会发现，安装完成之后，用node –v查看，还是老版本，安装未生效。

原因：
n 切换之后的 node 默认装在 /usr/local/bin/node，先用 which node 检查一下当前使用的 node 是否是这个路径下的。如上缘由，一般都是因为当前版本指定到了其他路径，更新下/etc/profile文件指定即可。轻松解决。