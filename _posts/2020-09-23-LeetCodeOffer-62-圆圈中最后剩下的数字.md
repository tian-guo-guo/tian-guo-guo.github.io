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

# [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

tag: easy，数组

**题目：**

0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

**示例1：**

```
输入: n = 5, m = 3
输出: 3
```

**示例1：**

```
输入: n = 10, m = 17
输出: 2
```

# 方法一：

**解题思路**

这个游戏里叫击鼓传花，西方也有叫热土豆。可通过严格的数学推导直接计算得出。但我们用程序来全真模拟才更真实。

1.  生成一个0、1、…、n-1的列表，初始索引i=0
2.  传递m次，意味着从i开始偏移m得到新索引i=i+m-1，考虑m可能大于当前列表长度，所以要对列表长度求模余
3.  从列表中pop出一个值后，实际上下一次偏移的初始索引仍然是当前pop掉的那个（因为后边的值都前移了一位），所以继续用i=i+m-1迭代新的索引，当然也要用新的列表长度求模余
4.  直至列表长度为1，返回最后剩下的数字。

由于列表要执行pop操作 n-1次，而每次pop(i)是平均O(n)复杂度，所以总的时间复杂度是O(n^2)

```python
class Solution:
    def lastRemaining(self, n, m):
        i, a = 0, list(range(n))
        while len(a) > 1:
            i = (i + m - 1) % len(a)
            a.pop(i)
        return a[0]
```

>执行用时：2200 ms, 在所有 Python3 提交中击败了14.03%的用户
>
>内存消耗：16.7 MB, 在所有 Python3 提交中击败了55.65%的用户

[Link](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/pythonquan-zhen-mo-ni-by-luanz/)