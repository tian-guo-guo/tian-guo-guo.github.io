---
layout:     post           # 使用的布局（不需要改）
title:      NPM Error gyp No Xcode or CLT version detected!
subtitle:   NPM Error gyp No Xcode or CLT version detected!  #副标题
date:       2021-1-18             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - vue
 
---

# NPM Error：gyp: No Xcode or CLT version detected!

 问题

最近在macOS Catalina中使用npm安装模块，经常会出现如下错误：

```bash
> node-gyp rebuild

No receipt for 'com.apple.pkg.CLTools_Executables' found at '/'.

No receipt for 'com.apple.pkg.DeveloperToolsCLILeo' found at '/'.

No receipt for 'com.apple.pkg.DeveloperToolsCLI' found at '/'.

gyp: No Xcode or CLT version detected!
gyp ERR! configure error 
gyp ERR! stack Error: `gyp` failed with exit code: 1
gyp ERR! stack     at ChildProcess.onCpExit (/usr/local/lib/node_modules/npm/node_modules/node-gyp/lib/configure.js:351:16)
gyp ERR! stack     at ChildProcess.emit (events.js:210:5)
gyp ERR! stack     at Process.ChildProcess._handle.onexit (internal/child_process.js:272:12)
gyp ERR! System Darwin 19.3.0
gyp ERR! command "/usr/local/bin/node" "/usr/local/lib/node_modules/npm/node_modules/node-gyp/bin/node-gyp.js" "rebuild"
gyp ERR! cwd /Users/yangjian/Documents/temp/test001/node_modules/fsevents
gyp ERR! node -v v12.13.0
gyp ERR! node-gyp -v v5.0.5
gyp ERR! not ok
```

-   截图如下



![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210118204636.jpg)



## 解决方案

### 1. 尝试用如下命令进行修复

```bash
$ xcode-select --install
```

系统提示如下信息：

```bash
xcode-select: error: command line tools are already installed, use "Software Update" to install updates
```

而事实上并没有所谓的"Software Update"可以更新

### 2. 正确姿势

一筹莫展之际，找到如下解决方案：

```bash
$ sudo rm -rf $(xcode-select -print-path)
$ xcode-select --install
```

>   请参见：
>   \- [https://github.com/schnerd/d3-scale-cluster/issues/7](https://link.zhihu.com/?target=https%3A//github.com/schnerd/d3-scale-cluster/issues/7)
>   \- [https://github.com/nodejs/node-gyp/blob/master/macOS_Catalina.md](https://link.zhihu.com/?target=https%3A//github.com/nodejs/node-gyp/blob/master/macOS_Catalina.md)
>
>   https://zhuanlan.zhihu.com/p/105526835