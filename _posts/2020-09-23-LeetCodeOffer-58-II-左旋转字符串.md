---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer58-II-左旋转字符串
subtitle:   LeetCode-Offer58-II-左旋转字符串 #副标题
date:       2020-09-23            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

tag: easy，字符串

**题目：**

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

**示例1：**

```
输入: s = "abcdefg", k = 2
输出: "cdefgab"
```

**示例2：**

```
输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
```

# 方法一：字符串切片

>   应用字符串切片函数，可方便实现左旋转字符串。

获取字符串 s[n:] 切片和 s[:n] 切片，使用 "+" 运算符拼接并返回即可。

**复杂度分析：**

- 时间复杂度 O(N) ： 其中 N 为字符串 s 的长度，字符串切片函数为线性时间复杂度（参考资料）；
- 空间复杂度 O(N) ： 两个字符串切片的总长度为 N 。

```python
class Solution:
    def reverseLeftWords(self, s, n):
        return s[n:] + s[:n]
```

>执行用时：36 ms, 在所有 Python3 提交中击败了89.65%的用户
>
>内存消耗：13.3 MB, 在所有 Python3 提交中击败了86.13%的用户

# 方法二：列表遍历拼接

>   若面试规定不允许使用 切片函数 ，则使用此方法。

**算法流程：**

1.  新建一个 list(Python)、StringBuilder(Java) ，记为 res ；
2.  先向 res 添加 “第 n+1 位至末位的字符” ；
3.  再向 res 添加 “首位至第 n 位的字符” ；
4.  将 res 转化为字符串并返回。

**复杂度分析**：

- 时间复杂度 O(N)O(N) ： 线性遍历 ss 并添加，使用线性时间；
- 空间复杂度 O(N) ： 新建的辅助 res 使用 O(N) 大小的额外空间。

```python
class Solution:
    def reverseLeftWords(self, s, n):
        res = []
        for i in range(n, len(s)):
            res.append(s[i])
        for i in range(n):
            res.append(s[i])
        return ''.join(res)
```

# 方法三：字符串遍历拼接

>   若规定 Python 不能使用 join() 函数，或规定 Java 只能用 String ，则使用此方法。

此方法与 方法二 思路一致，区别是使用字符串代替列表。

**复杂度分析：**

-   时间复杂度 O(N)： 线性遍历 s 并添加，使用线性时间；
-   空间复杂度 O(N)： 假设循环过程中内存会被及时回收，内存中至少同时存在长度为 N 和 N−1 的两个字符串（新建长度为 N 的 res 需要使用前一个长度 N−1 的 res ），因此至少使用 O(N) 的额外空间。

```python
class Solution:
    def reverseLeftWords(self, s, n):
        res = ""
        for i in range(n, len(s)):
            res += s[i]
        for i in range(n):
            res += s[i]
        return res
```

