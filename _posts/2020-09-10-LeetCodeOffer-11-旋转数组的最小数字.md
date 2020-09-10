---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer11-旋转数组的最小数字
subtitle:   LeetCode-Offer11-旋转数组的最小数字 #副标题
date:       2020-09-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

tag: easy，数组

**题目：**

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

**示例 1：**

```
输入：[3,4,5,1,2]
输出：1
```

**示例 2：**

```
输入：[2,2,2,0,1]
输出：0
```

# 方法一：二分法

**二分法** 解决，其可将 **遍历法** 的 **线性级别** 时间复杂度降低至 **对数级别**。

- 初始化： 声明 i, j 双指针分别指向 nums数组左右两端；
- 循环二分： 设 m = (i + j) / 2为每次二分的中点（ "/" 代表向下取整除法，因此恒有 i≤m<j ），可分为以下三种情况：
    - 当 nums[m] > nums[j] 时： m 一定在 左排序数组 中，即旋转点 x 一定在 [m + 1, j][m+1,j] 闭区间内，因此执行i=m+1；
    - 当 nums[m]<nums[j] 时： m 一定在 右排序数组 中，即旋转点 x 一定在[i, m][i,m] 闭区间内，因此执行j=m；
    - 当 nums[m]=nums[j] 时： 无法判断 m 在哪个排序数组中，即无法判断旋转点 x 在 [i, m][i,m] 还是 [m + 1, j][m+1,j] 区间中。解决方案： 执行j=j−1 缩小判断范围，分析见下文。
- 返回值： 当i=j 时跳出二分循环，并返回 旋转点的值 nums[i] 即可。

```python
class Solution:
    def minArray(self, numbers):
        i, j = 0, len(numbers) - 1
        while i < j:
            m = (i + j ) // 2
            if numbers[m] > numbers[j]:
                i = m + 1
            elif numbers[m] < numbers[j]:
                j = m
            else:
                j -= 1
        return numbers[i]
```

-   时间复杂度 O(log 2 N) ： 在特例情况下（例如 [1, 1, 1, 1][1,1,1,1]），会退化到 O(N)O(N)。
-   空间复杂度 O(1) ： i , j , m 变量使用常数大小的额外空间。

>执行用时：32 ms, 在所有 Python3 提交中击败了98.03%的用户
>
>内存消耗：13.9 MB, 在所有 Python3 提交中击败了45.40%的用户

[Link](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/solution/mian-shi-ti-11-xuan-zhuan-shu-zu-de-zui-xiao-shu-3/)

