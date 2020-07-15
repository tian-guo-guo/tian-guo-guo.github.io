---
layout:     post                    # 使用的布局（不需要改）
title:      CUDA之nvidia-smi命令详解              # 标题 
subtitle:   CUDA之nvidia-smi命令详解 #副标题
date:       2020-05-18              # 时间
author:     甜果果                      # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-keybord.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 学习日记
    - 服务器

---

# [CUDA之nvidia-smi命令详解](https://blog.csdn.net/Bruce_0712/article/details/63683787)

![nvidia-smi截图](https://tva1.sinaimg.cn/large/007S8ZIlly1gewylryi6pj30k50ecgnm.jpg)

![image-20200518221615250](https://tva1.sinaimg.cn/large/007S8ZIlly1gewypj8c1oj30u00utjwt.jpg)



上面的表格中：

-   第一栏的Fan：N/A是风扇转速，从0到100%之间变动，这个速度是计算机期望的风扇转速，实际情况下如果风扇堵转，可能打不到显示的转速。有的设备不会返回转速，因为它不依赖风扇冷却而是通过其他外设保持低温（比如我们实验室的服务器是常年放在空调房间里的）。

-   第二栏的Temp：是温度，单位摄氏度。

-   第三栏的Perf：是性能状态，从P0到P12，P0表示最大性能，P12表示状态最小性能。

-   第四栏下方的Pwr：是能耗，上方的Persistence-M：是持续模式的状态，持续模式虽然耗能大，但是在新的GPU应用启动时，花费的时间更少，这里显示的是off的状态。

-   第五栏的Bus-Id是涉及GPU总线的东西，domain: bus:device.function

-   第六栏的Disp.A是Display Active，表示GPU的显示是否初始化。

    第五第六栏下方的Memory Usage是显存使用率。

-   第七栏是浮动的GPU利用率。

-   第八栏上方是关于ECC的东西。

    第八栏下方Compute M是计算模式。

    

下面一张表示每个进程占用的显存使用率。

-   **显存占用和GPU占用是两个不一样的东西**，显卡是由GPU和显存等组成的，显存和GPU的关系有点类似于内存和CPU的关系。我跑caffe代码的时候显存占得少，GPU占得多，师弟跑TensorFlow代码的时候，显存占得多，GPU占得少。

