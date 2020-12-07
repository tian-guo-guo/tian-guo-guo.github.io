---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer42-连续子数组的最大和
subtitle:   LeetCode-Offer42-连续子数组的最大和 #副标题
date:       2020-09-20            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/) [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

tag: easy，数组，动态规划

**题目：**

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

 **示例一**：

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

# 方法一：动态规划

1.连续子数组的最大和,先分析两种极端情况：
[1]数组全为负数 如 nums = [-1，-2，-3，-4，-5] 连续子数组的最大和为 nums[0]
[2]数组中存在一个正数 如 nums = [-1，-2，1，-4，-5] 连续子数组的最大和为 nums[2] 1
可以看出要想获得最大连续的子数组和，必须添加进去的数字为正数
2.很容易想到动态规划：
[1]初始条件 maxnum = nums[0]
[2]开始循环遍历
[3]当数值大于零的时候 nums[i-1] >0 ，需要添加进求和项,直接在元素组上修改 nums[i] += last_num
[4]状态转移方程： maxnum = max(maxnum, nums[i]) 判断上一个的连续子数组求和最大值与当前的大小关系

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxnum = nums[0]
        for i in range(1,len(nums)):
            last_num = nums[i-1]
            if last_num > 0:
                nums[i] += last_num
            maxnum = max(maxnum, nums[i])
        return maxnum
```

[Link](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/solution/si-lu-jian-dan-xing-neng-jie-jin-70-by-jamleon/)



我们在原地修改数组，将数组每个位置的值更改为当前位置上的最大和。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 动态规划，原地修改数组
        maxnum = nums[0]
        for i in range(1,len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            maxnum = max(maxnum,nums[i])
        return maxnum
```

\- 时间复杂度：O(N)，遍历了一次数组。
\- 空间复杂度：O(1)，使用了常数空间。

>执行用时：72 ms, 在所有 Python3 提交中击败了86.93%的用户
>
>内存消耗：17.7 MB, 在所有 Python3 提交中击败了68.11%的用户

[Link](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/solution/dong-tai-gui-hua-by-ml-zimingmeng-2/)