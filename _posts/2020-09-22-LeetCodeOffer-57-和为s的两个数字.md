---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer55-II-平衡二叉树110平衡二叉树
subtitle:   LeetCode-Offer55-II-平衡二叉树110平衡二叉树 #副标题
date:       2020-09-22            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

tag: easy，数组，双指针

**题目：**

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

**示例1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```

**示例2：**

```
输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
```

# 方法一：

**解题思路：**
利用 HashMap 可以通过遍历数组找到数字组合，时间和空间复杂度均为 O(N)；
注意本题的 nums 是 排序数组 ，因此可使用 双指针法 将空间复杂度降低至 O(1) 。

**算法流程：**

1.  初始化： 双指针 i , j 分别指向数组 nums 的左右两端 （俗称对撞双指针）。
2.  循环搜索： 当双指针相遇时跳出；
    1.  计算和 s = nums[i] + nums[j] ；
    2.  若 s > target ，则指针 j 向左移动，即执行 j = j - 1 ；
    3.  若 s < target ，则指针 i 向右移动，即执行 i = i + 1 ；
    4.  若 s = target ，立即返回数组 [nums[i], nums[j]] ；
3.  返回空数组，代表无和为 target 的数字组合。

**复杂度分析：**

- 时间复杂度 O(N) ： NN 为数组 nums 的长度；双指针共同线性遍历整个数组。
- 空间复杂度 O(1) ： 变量 i, j 使用常数大小的额外空间。

```python
class Solution:
    def twoSum(self, nums, target):
        i, j = 0, len(nums) - 1
        while i < j:
            s = nums[i] + nums[j]
            if s > target: 
                j -= 1
            elif s < target: 
                i += 1
            else: 
                return nums[i], nums[j]
        return []
```

>执行用时：144 ms, 在所有 Python3 提交中击败了81.77%的用户
>
>内存消耗：24.3 MB, 在所有 Python3 提交中击败了80.81%的用户

[Link](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/solution/mian-shi-ti-57-he-wei-s-de-liang-ge-shu-zi-shuang-/)

