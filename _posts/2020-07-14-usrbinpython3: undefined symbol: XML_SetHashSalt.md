---
layout:     post           # 使用的布局（不需要改）
title:      /usr/bin/python3 undefined symbol XML_SetHashSalt
subtitle:   undefined symbol XML_SetHashSalt #副标题
date:       2020-07-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
 
---

# /usr/bin/python3: undefined symbol: XML_SetHashSalt

启动tensorboard，以及使用pip安装库，都会报错undefined symbol: XML_SetHashSalt

```
(fairseq) root@b09078bbef4b:~/tian/fairseq# tensorboard --logdir=ne_WIPO_MT-transformer
/usr/bin/python3: symbol lookup error: /usr/bin/python3: undefined symbol: XML_SetHashSalt
```

后来在网上查了查发现是因为改变了LD_LIBRARY_PATH的值。其值现在是：

```
(fairseq) root@b09078bbef4b:~/tian/fairseq# echo $LD_LIBRARY_PATH
:/usr/local/cuda-10.1/lib64
```

具体原因如下：

执行命令 ldd /usr/lib/x86_64-linux-gnu/libpython3.5m.so.1.0，得到结果如下：

```
(fairseq) root@b09078bbef4b:~/tian/fairseq# ldd /usr/lib/x86_64-linux-gnu/libpython3.5m.so.1.0
        linux-vdso.so.1 =>  (0x00007ffe5dfd5000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f65870a9000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f6586cdf000)
        libexpat.so.1 => /usr/local/lib/libexpat.so.1 (0x00007f6586ab6000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f658689c000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f6586698000)
        libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007f6586495000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f658618c000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f658794d000)
```

从上图中我们发现libexpat.so.1的路径更改了。本来应该使用系统中的libexpat.so.1，其路径如下图所示：

```
(fairseq) root@b09078bbef4b:/usr/local/lib# ldd /usr/lib/x86_64-linux-gnu/libpython3.5m.so.1.0
        linux-vdso.so.1 =>  (0x00007ffeaa59b000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f081c2f4000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f081bf2a000)
        libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007f081bd01000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f081bae7000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f081b8e3000)
        libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007f081b6e0000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f081b3d7000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f081cb98000)
```

解决办法：

方法一：直接将/usr/local/lib/libexpat.so.1 文件改名，这样子就能将其隐藏。改名命令如下：

  mv libexpat.so.1 libexpat.so.1.NOFIND

这种办法比较粗暴总感觉会有影响，虽然我刚开始用的就是这种做法，而且还没发现有什么问题~~

 方法二：上面也说了出现这个问题的原因是动态库调错了。这是由于在安装oracle时设置LD_LIBRARY_PATH设置的有问题，我直接设置成了oracle安装路径下的lib，如下：

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200714195712.png)

需要将其修改成如下：

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200714195718.png)

这里得说明两点：

1）上面设置的其实就是系统在调用链接库的时候，可以从/lib、/lib/x86_64-linux-gnu、/home/cjh/lib、/usr/lib、/usr/lib:/home/cjh/tools/oracle11g/product/11.2.0/dbhome_1/lib这些路径下查找(cjh是我的用户名，这里我尽量把系统中有lib的目录都包含了)。我一开始没加/lib/x86_64-linux-gnu，以为包含了/lib路径就行了，然而没有起作用。这可能和系统在选择先遍历那个路径的方式有关，所以尽量把路径写到文件所在路径，而且要写在oralce的lib之前(写在后面亲测有没效果)。

2）我修改的是用户下的.bashrc文件，因为我安装oracle时就是在这个文件里面设置的。如果你是在其他文件设置的，可以根据自己的实际情况进行修改。对于.bashrc、.profile、.bash_profile、profile之间的区别可以参考这篇文章：

https://blog.csdn.net/gatieme/article/details/45064705。记住修改完之后要执行 source ~/.bashrc,让bash重新读取这个文件，而且这个命令只对一个终端有用，每一个终端都要执行，你可以重新打开终端。

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200714195732.png)

===》》》》网上的一些解释：

https://ubuntuforums.org/showthread.php?t=2094005

https://bbs.archlinux.org/viewtopic.php?id=140916

https://bugzilla.redhat.com/show_bug.cgi?id=821337

[原文链接](https://blog.csdn.net/J_H_C/article/details/84961219)

