---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer50-第一个只出现一次的字符
subtitle:   LeetCode-Offer50-第一个只出现一次的字符 #副标题
date:       2020-09-20            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

tag: easy，字符串，哈希表

**题目：**

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

 **示例一**：

```
s = "abaccdeff"
返回 "b"

s = "" 
返回 " "
```

# 方法一：哈希表

**思路：**

1.  遍历字符串 `s` ，使用哈希表统计 “各字符数量是否 > 1 ”。
2.  再遍历字符串 `s` ，在哈希表中找到首个 “数量为 1 的字符”，并返回。

**算法流程：**

1.  初始化： 字典 (Python)、HashMap(Java)、map(C++)，记为 dic ；
2.  字符统计： 遍历字符串 s 中的每个字符 c ；
    1.  若 dic 中 不包含 键(key) c ：则向 dic 中添加键值对 (c, True) ，代表字符 c 的数量为 1 ；
    2.  若 dic 中 包含 键(key) c ：则修改键 c 的键值对为 (c, False) ，代表字符 c 的数量 > 1>1 。
3.  查找数量为 11 的字符： 遍历字符串 s 中的每个字符 c ；
    1.  若 dic中键 c 对应的值为 True ：，则返回 c 。
4.  返回 ' ' ，代表字符串无数量为 1 的字符。

**复杂度分析：**

- 时间复杂度 O(N) ： N 为字符串 s 的长度；需遍历 s 两轮，使用 O(N) ；HashMap 查找操作的复杂度为 O(1) ；
- 空间复杂度 O(1) ： 由于题目指出 s 只包含小写字母，因此最多有 26 个不同字符，HashMap 存储需占用 O(26) = O(1) 的额外空间。

```python
class Solution:
    def firstUniqChar(self, s):
        dic = {}
        for c in s:
            dic[c] = not c in dic
        for c in s:
            if dic[c]: return c
        return ' '
```

>执行用时：92 ms, 在所有 Python3 提交中击败了79.51%的用户
>
>内存消耗：13.5 MB, 在所有 Python3 提交中击败了48.75%的用户

# 方法二：有序哈希表

**思想：**

在哈希表的基础上，有序哈希表中的键值对是 按照插入顺序排序 的。基于此，可通过遍历有序哈希表，实现搜索首个 “数量为 11 的字符”。

哈希表是 去重 的，即哈希表中键值对数量 \leq≤ 字符串 s 的长度。因此，相比于方法一，方法二减少了第二轮遍历的循环次数。当字符串很长（重复字符很多）时，方法二则效率更高。

**复杂度分析：**

时间和空间复杂度均与 “方法一” 相同，而具体分析：方法一 需遍历 `s` 两轮；方法二 遍历 `s` 一轮，遍历 `dic` 一轮（ `dic` 的长度不大于 26 ）。

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = collections.OrderedDict()
        for c in s:
            dic[c] = not c in dic
        for k, v in dic.items():
            if v: return k
        return ' '
```

[Link](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/solution/mian-shi-ti-50-di-yi-ge-zhi-chu-xian-yi-ci-de-zi-3/)