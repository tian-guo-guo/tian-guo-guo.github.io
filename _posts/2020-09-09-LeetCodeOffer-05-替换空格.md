---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer05-替换空格
subtitle:   LeetCode-Offer05-替换空格 #副标题
date:       2020-09-09            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

tag: easy，字符串

**题目：**

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

**示例1：**

```
输入：s = "We are happy."
输出："We%20are%20happy."
```

# 方法一：

新建一个空字符串，遍历给定的字符串，如果是空格就替换成%20，不是的话加到到新建的字符串后面

```python
class Solution:
    def replaceSpace(self, str):
        string = ''
        for i in str:
            if i == " ":
                string += '%20'
            else:
                string += i
        return string
```

-   时间复杂度 O(N) ：遍历字符串。
-   空间复杂度 O(N) ：字符串的长度。

>执行用时：40 ms, 在所有 Python3 提交中击败了66.03%的用户
>
>内存消耗：13.7 MB, 在所有 Python3 提交中击败了29.50%的用户

# 方法二：

简单一点的写法，用''.join()函数直接返回结果

```python
class Solution:
    def replaceSpace(self, str):
        return ''.join(('%20' if i==' ' else i for i in str))
```

先结果，再判断，最后遍历

>执行用时：44 ms, 在所有 Python3 提交中击败了40.06%的用户
>
>内存消耗：13.7 MB, 在所有 Python3 提交中击败了41.55%的用户

# 方法三：

直接用replace函数

```python
class Solution:
    def replaceSpace(self, str):
        return str.replace(' ', '%20')
```

>执行用时：40 ms, 在所有 Python3 提交中击败了66.03%的用户
>
>内存消耗：13.7 MB, 在所有 Python3 提交中击败了45.03%的用户

[Link](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/submissions/)

