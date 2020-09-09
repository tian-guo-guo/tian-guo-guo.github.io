---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer03-数组中重复的数字
subtitle:   LeetCode-Offer03-数组中重复的数字 #副标题
date:       2020-09-09            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

tag: easy，哈希表，数组

**题目：**

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**示例1：**

```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

# 方法一 使用集合：

一、使用集合

遍历数组，判断nums[i]是否存在于集合中，如果存在就返回；不存在就添加到集合中。

```python
class Solution:
    def findRepeatNumber(self, nums):
        count = set()
        for num in nums:
            if num in count:
                return num
            else:
                count.add(num)
```

>执行用时：76 ms, 在所有 Python3 提交中击败了11.73%的用户
>
>内存消耗：22.9 MB, 在所有 Python3 提交中击败了96.79%的用户

-   时间复杂度：O(N)，最坏情况，循环遍历完了整个数组
-   空间复杂度：O(N)，哈希表。

二、使用哈希表（字典）

建立一个字典，以数字当做唯一的键，索引当做值，循环遍历数组里的每一个数，如果字典中已经存在这个数字，返回这个数字，不然的话就添加这个数字到字典中。

```python
class Solution:
    def findRepeatNumber(self, nums):
        dic = {}
        for index, num in enumerate(nums):
            if num in dic:
                return num
            else:
                dic[num] = index
```

>执行用时：80 ms, 在所有 Python3 提交中击败了8.41%的用户
>
>内存消耗：23.1 MB, 在所有 Python3 提交中击败了62.17%的用户

# 方法二：排序

对数组进行排序，相同的数字会碰到一起，从1开始遍历数组，如果当前数与上一个数是相等的，就返回这个数。

```python
class Solution:
    def findRepeatNumber(self, nums):
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i -1]:
                return nums[i]
```

>执行用时：64 ms, 在所有 Python3 提交中击败了34.06%的用户
>
>内存消耗：23.2 MB, 在所有 Python3 提交中击败了32.08%的用户

-   时间复杂度：O(NlogN)，对数组进行排序
-   空间复杂度：O(1)

[Link](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/solution/jian-ji-ha-xi-biao-by-ml-zimingmeng/)

