---
layout:     post           # 使用的布局（不需要改）
title:      Python_inf一些用法
subtitle:   Python_inf一些用法 #副标题
date:       2020-07-06            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-ios10.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - python


---

# Python_inf一些用法

Python中可以用如下方式表示正负无穷：

```
float("inf"), float("-inf")
```

利用 inf 做简单加、乘算术运算仍会得到 inf

```
>>> 1 + float('inf')
inf
>>> 2 * float('inf')
inf
```

 

但是利用 inf 乘以0会得到 not-a-number(NaN)：

```
>>> 0 * float("inf")
nan
```

除了inf外的其他数除以inf，会得到0

```
>>> 889 / float('inf')
0.0
>>> float('inf')/float('inf')
nan
```

 

通常的运算是不会得到 inf值的 

```
>>> 2.0**2
4.0
>>> _**2
16.0
>>> _**2
256.0
>>> _**2
65536.0
>>> _**2
4294967296.0
>>> _**2
1.8446744073709552e+19
>>> _**2
3.4028236692093846e+38
>>> _**2
1.157920892373162e+77
>>> _**2
1.3407807929942597e+154
>>> _**2
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
OverflowError: (34, 'Numerical result out of range')
```

inf的运算规则遵从 [IEEE-754 standard](http://en.wikipedia.org/wiki/IEEE_754-1985)

**不等式：**

当涉及 > 和 < 运算时，

-   所有数都比-inf大
-   所有数都比+inf小

 **等式：**

+inf 和 +inf相等

-inf 和 -inf相等

[Link](https://blog.csdn.net/SHENNONGZHAIZHU/article/details/51997887)

