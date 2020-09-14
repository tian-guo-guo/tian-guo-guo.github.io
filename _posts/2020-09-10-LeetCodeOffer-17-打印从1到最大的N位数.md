---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer17-打印从1到最大的n位数
subtitle:   LeetCode-Offer17-打印从1到最大的n位数 #副标题
date:       2020-09-14            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

tag: easy，字符串，分治算法

**题目：**

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

示例 1：

```
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

# 方法一：

1.  最大的 nn 位数（记为 endend ）和位数 nn 的关系： 例如最大的 11 位数是 99 ，最大的 22 位数是 9999 ，最大的 33 位数是 999999 。则可推出公式：end = 10^n - 1
2.  大数越界问题： 当 n 较大时，end 会超出 int32 整型的取值范围，超出取值范围的数字无法正常存储。但由于本题要求返回 int 类型数组，相当于默认所有数字都在 int32 整型取值范围内，因此不考虑大数越界问题。

因此，只需定义区间 [1, 10^n - 1][1,10 n −1] 和步长 11 ，通过 forfor 循环生成结果列表 resres 并返回即可。

-   复杂度分析：
    -   时间复杂度 O(10^n)： 生成长度为 10^n的列表需使用 O(10^n)时间。
    -   空间复杂度 O(1) ： 建立列表需使用 O(1) 大小的额外空间（ 列表作为返回结果，不计入额外空间 ）。

```python
class Solution:
    def printNumbers(self, n):
        res = []
        for i in range(1, 10 ** n):
            res.append(i)
        return res
```

换一种写法

```python
class Solution:
    def printNumbers(self, n):
        return list(range(1, n ** 10))
```

>执行用时：48 ms, 在所有 Python3 提交中击败了67.79%的用户
>
>内存消耗：19.4 MB, 在所有 Python3 提交中击败了56.48%的用户

