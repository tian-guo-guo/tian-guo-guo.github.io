---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer10-1-斐波那契数列
subtitle:   LeetCode-Offer10-1-斐波那契数列 #副标题
date:       2020-09-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

tag: easy，递归，动态规划

**题目：**

写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项。斐波那契数列的定义如下：例如，给出

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：1
```

**示例 2：**

```
输入：n = 5
输出：5
```

# 方法一：动态规划

斐波那契数列的定义是 f(n + 1) = f(n) + f(n - 1) ，生成第 n项的做法有以下几种：

- 递归法：
    - 原理： 把 f(n)问题的计算拆分成 f(n-1)和 f(n-2)两个子问题的计算，并递归，以 f(0)和 f(1)为终止条件。
    - 缺点： 大量重复的递归计算，例如 f(n)和 f(n - 1)两者向下递归需要 各自计算 f(n - 2)的值。
- 记忆化递归法：
    - 原理： 在递归法的基础上，新建一个长度为 n的数组，用于在递归时存储 f(0)至 f(n)的数字值，重复遇到某数字则直接从数组取用，避免了重复的递归计算。
    - 缺点： 记忆化存储需要使用 O(N)的额外空间。
- 动态规划：
    - 原理： 以斐波那契数列性质 f(n + 1) = f(n) + f(n - 1)为转移方程。
    - 从计算效率、空间复杂度上看，动态规划是本题的最佳解法。

动态规划解析：

-   状态定义： 设 dp为一维数组，其中 dp[i]的值代表 斐波那契数列第 i个数字 。
-   转移方程： dp[i + 1] = dp[i] + dp[i - 1] ，即对应数列定义 f(n + 1) = f(n) + f(n - 1) ；
-   初始状态： dp[0] = 0, dp[1] = 1 ，即初始化前两个数字；
-   返回值： dp[n]，即斐波那契数列的第 n 个数字。

```python
class Solution:
    def fib(self, n):
        f1 = 0
        f2 = 1
        for _ in range(n):
            f1, f2 = f2, f1+f2
        return f1 % 1000000007
```

-   **时间复杂度 O(N)：** 计算 f(n) 需循环 n次，每轮循环内计算操作使用 O(1)。
-   **空间复杂度 O(N)：** 几个标志变量使用常数大小的额外空间。

>执行用时：36 ms, 在所有 Python3 提交中击败了88.27%的用户
>
>内存消耗：13.5 MB, 在所有 Python3 提交中击败了88.36%的用户

[Link](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/solution/mian-shi-ti-10-i-fei-bo-na-qi-shu-lie-dong-tai-gui/)



```python
class Solution:
    def fib(self, n: int) -> int:
####标准递归解法：
        if n==0:return 0
        if n==1:return 1
        return (self.fib(n-1)+self.fib(n-2))%1000000007
####带备忘录的递归解法
        records = [-1 for i in range(n+1)] # 记录计算的值
        if n == 0:return 0
        if n == 1:return 1
        if records[n] == -1: # 表明这个值没有算过
            records[n] = self.fib(n-1) +self.fib(n-2)
        return records[n]
#递归输出超时,用记忆化递归规划，时间上优越很多。

###DP方法：解决记忆化递归费内存的问题
        dp={}
        dp[0]=0
        dp[1]=1
        if n>=2:
            for i in range(2,n+1):
                dp[i]=dp[i-1]+dp[i-2]
        return dp[n]%1000000007


###最优化DP方法：
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007

```

