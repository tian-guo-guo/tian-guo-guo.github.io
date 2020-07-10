---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-28-实现strStr()
subtitle:   LeetCode-28-实现strStr() #副标题
date:       2020-07-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python

---

# LeetCode-[28. 实现 strStr](https://leetcode-cn.com/problems/implement-strstr/)

<iframe width="560" height="315" src="https://www.youtube.com/embed/62lXzTIHTiI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

tag: easy，字符串

**题目：**

实现 strStr() 函数。

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

**示例1：**

```
输入: haystack = "hello", needle = "ll"
输出: 2
```

**示例2：**

```
输入: haystack = "aaaaa", needle = "bba"
输出: -1
```

**说明:**

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

# 方法一：

先写for loop，i不会是从第一开始到结尾，因为...，range应该是len(haystack)-len(needle)+1，

然后两两的找有没有跟needle相同的排序，每次从haystack提出两个去做比较，如果存在就返回i，

i就是needle起始的位置，

```python
class Solution:
    def strStr(self, haystack, needle):
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i 
        return -1
```

>执行用时：44 ms, 在所有 Python3 提交中击败了61.81%的用户
>
>内存消耗：13.9 MB, 在所有 Python3 提交中击败了6.67%的用户