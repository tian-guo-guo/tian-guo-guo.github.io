---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer14-1-剪绳子 343 整数拆分
subtitle:   LeetCode-Offer14-1-剪绳子 343 整数拆分 #副标题
date:       2020-09-14            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/) [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)

tag: medium

**题目：**

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

示例 1：

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

示例 2：

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

# 方法一：

解题思路：

- 设将长度为 n*n* 的绳子切为 a* 段：

*n*=*n*1+*n*2+...+*n**a*

- 本题等价于求解：

max(*n*1×*n*2×...×*n**a*)

>   以下数学推导总体分为两步：① 当所有绳段长度相等时，乘积最大。② 最优的绳段长度为 3 。

![image-20200914083206546](/Users/suntian/Library/Application Support/typora-user-images/image-20200914083206546.png)

```python
class Solution:
    def cuttingRope(self, n):
        if n <= 3:
            return n - 1
        a, b = n // 3, n % 3
        if b == 0:
            return int(math.pow(3, a))
        if b == 1:
            return int(math.pow(3, a - 1) * 4)
        return int(math.pow(3, a) * 2)
```

>执行用时：36 ms, 在所有 Python3 提交中击败了92.83%的用户
>
>内存消耗：13.7 MB, 在所有 Python3 提交中击败了45.45%的用户

