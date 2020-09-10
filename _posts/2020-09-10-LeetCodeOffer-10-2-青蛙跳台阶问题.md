---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer10-2-青蛙跳台阶问题
subtitle:   LeetCode-Offer10-2-青蛙跳台阶问题 #副标题
date:       2020-09-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

tag: easy，动态规划

**题目：**

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```
输入：n = 2
输出：2
```

**示例 2：**

```
输入：n = 7
输出：21
```

**示例 3：**

```
输入：n = 0
输出：1
```

# 方法一：动态规划

>此类求 多少种可能性 的题目一般都有 递推性质 ，即 f(n) 和 f(n-1)…f(1)之间是有联系的。

-   设跳上 nn 级台阶有 f(n)种跳法。在所有跳法中，青蛙的最后一步只有两种情况： 跳上 1 级或 2 级台阶。
    -   当为 1 级台阶： 剩 n-1 个台阶，此情况共有 f(n-1)种跳法；
    -   当为 2 级台阶： 剩 n-2 个台阶，此情况共有 f(n-2)种跳法。
-   f(n)为以上两种情况之和，即 f(n)=f(n-1)+f(n-2) ，以上递推性质为斐波那契数列。本题可转化为 求斐波那契数列第 n 项的值 ，与 面试题10- I. 斐波那契数列 等价，唯一的不同在于起始数字不同。
    -   青蛙跳台阶问题： f(0)=1 , f(1)=1 , f(2)=2；
    -   斐波那契数列问题： f(0)=0 , f(1)=1 , f(2)=1 。

```python
class Solution:
    def numWays(self, n):
        f1 = 1
        f2 = 1
        for _ in range(n):
            f1, f2 = f2, f1+f2
        return f1 % 1000000007
```

-   **时间复杂度 O(N)：** 计算 f(n) 需循环 n次，每轮循环内计算操作使用 O(1)。
-   **空间复杂度 O(N)：** 几个标志变量使用常数大小的额外空间。

>执行用时：36 ms, 在所有 Python3 提交中击败了86.47%的用户
>
>内存消耗：13.8 MB, 在所有 Python3 提交中击败了6.38%的用户

[Link](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/solution/mian-shi-ti-10-ii-qing-wa-tiao-tai-jie-wen-ti-dong/)

