---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer21-调整数组顺序使奇数位于偶数前面
subtitle:   LeetCode-Offer21-调整数组顺序使奇数位于偶数前面 #副标题
date:       2020-09-15            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

tag: easy，数组，双指针

**题目：**

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

**示例 1：**

```
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
```

# 方法一：

-   解题思路：

考虑定义双指针 i , j 分列数组左右两端，循环执行：

1.  指针 i 从左向右寻找偶数；
2.  指针 jj 从右向左寻找奇数；
3.  将 偶数 nums[i] 和 奇数 nums[j] 交换。

可始终保证： 指针 i 左边都是奇数，指针 j 右边都是偶数 。

- 算法流程：

-   初始化： i , j 双指针，分别指向数组 nums 左右两端；
-   循环交换： 当 i = j时跳出；
    -   指针 i 遇到奇数则执行 i=i+1 跳过，直到找到偶数；
    -   指针 jj 遇到偶数则执行 j=j−1 跳过，直到找到奇数；
    -   交换 nums[i] 和 nums[j] 值；
-   返回值： 返回已修改的 numsnums 数组。

- 复杂度分析：
    - 时间复杂度 O(N) ： N 为数组 nums 长度，双指针 i, j 共同遍历整个数组。
    - 空间复杂度 O(1) ： 双指针 i, j 使用常数大小的额外空间。

```python
class Solution:
    def exchange(self, nums):
        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] & 1 == 1:
                i += 1
            while i < j and nums[j] & 1 == 0:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        return nums
```

>执行用时：60 ms, 在所有 Python3 提交中击败了66.40%的用户
>
>内存消耗：17.7 MB, 在所有 Python3 提交中击败了99.18%的用户

[Link](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/solution/mian-shi-ti-21-diao-zheng-shu-zu-shun-xu-shi-qi-4/)