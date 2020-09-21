---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer53-II-0～n-1中缺失的数字
subtitle:   LeetCode-Offer53-II-0～n-1中缺失的数字 #副标题
date:       2020-09-21            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

tag: easy，数组，二分查找

**题目：**

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

**示例1：**

```
输入: [0,1,3]
输出: 2
```

**示例2：**

```
输入: [0,1,2,3,4,5,6,7,9]
输出: 8
```

# 方法一：

**解题思路：**

- 排序数组中的搜索问题，首先想到 二分法 解决。
- 根据题意，数组可以按照以下规则划分为两部分。
    - 左子数组： nums[i] = i ；
    - 右子数组： nums[i] ≠ i；
- 缺失的数字等于 “右子数组的首位元素” 对应的索引；因此考虑使用二分法查找 “右子数组的首位元素” 。

**算法解析：**

1.  初始化： 左边界 i = 0，右边界 j = len(nums) - 1 ；代表闭区间 [i, j][i,j] 。
2.  循环二分： 当 i≤j 时循环 （即当闭区间 [i, j][i,j] 为空时跳出） ；
    1.  计算中点 m = (i + j) // 2 ，其中 "//" 为向下取整除法；
    2.  若 nums[m] = m ，则 “右子数组的首位元素” 一定在闭区间 [m + 1, j][m+1,j] 中，因此执行 i = m + 1；
    3.  若 nums[m] ≠ m，则 “左子数组的末位元素” 一定在闭区间 [i, m - 1][i,m−1] 中，因此执行 j = m - 1；
        返回值： 跳出时，变量 i 和 j 分别指向 “右子数组的首位元素” 和 “左子数组的末位元素” 。因此返回 i 即可。

##### 复杂度分析：

-   **时间复杂度 O(log N)：** 二分法为对数级别复杂度。
-   **空间复杂度 O(1)：** 几个变量使用常数大小的额外空间。

```python
class Solution:
    def missingNumber(self, nums):
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] == m: 
                i = m + 1
            else: 
                j = m - 1
        return i
```

>执行用时：44 ms, 在所有 Python3 提交中击败了78.79%的用户
>
>内存消耗：14.3 MB, 在所有 Python3 提交中击败了49.87%的用户

[Link](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/solution/mian-shi-ti-53-ii-0n-1zhong-que-shi-de-shu-zi-er-f/)

