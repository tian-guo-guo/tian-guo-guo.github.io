---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer59-I-滑动窗口的最大值239
subtitle:   LeetCode-Offer59-I-滑动窗口的最大值239 #副标题
date:       2020-09-23            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

tag: easy，数组，哈希表，排序

**题目：**

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

**示例1：**

```
输入: [1,2,3,4,5]
输出: True
```

**示例1：**

```
输入: [0,0,1,2,5]
输出: True
```

# 方法一：

