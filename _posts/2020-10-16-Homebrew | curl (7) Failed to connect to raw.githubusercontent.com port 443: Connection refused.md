---
layout:     post           # 使用的布局（不需要改）
title:      Homebrew | curl (7) Failed to connect to raw.githubusercontent.com port 443 Connection refused           # 标题 
subtitle:   Homebrew | curl (7) Failed to connect to raw.githubusercontent.com port 443 Connection refused #副标题
date:       2020-10-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [Homebrew | curl: (7) Failed to connect to raw.githubusercontent.com port 443: Connection refused](https://blog.csdn.net/u012400885/article/details/103849472)

某天想玩玩 Homebrew，突然提示如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200106003102790.jpeg)
依稀记得，我曾经玩过这个东西啊，啥情况？

果断官网准备安装下 Homebrew：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200106003253981.png)
出师不利，自然的打开 Stack Overflow 查找解决之道：

-   [Homebrew installation on Mac OS X Failed to connect to raw.githubusercontent.com port 443](https://stackoverflow.com/questions/29910217/homebrew-installation-on-mac-os-x-failed-to-connect-to-raw-githubusercontent-com)

关键靠谱答案截图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200106003451943.png)
我们按照此步骤操作一番～

**Step 1：首先打开 Homebrew install：**

-   [Homebrew install](https://raw.githubusercontent.com/Homebrew/install/master/install)

将文件另存为：**brew_install.rb**

鉴于个人也出现了几次打不开的情况，这里附上百度网盘地址，方便小伙伴操作：

-   链接: https://pan.baidu.com/s/1HvJZj0dl9fDqtgzmVtK2iw 密码:umqb

**Step 2：ruby 安装已下载的 brew_install.rb：**

iTerm 2 键入如下指令：

-   ruby [brew_install.rb 地址]

如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200106003957981.png)
安装成功信息如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200106004122124.png)
哦可，玩去吧～