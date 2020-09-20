---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer53-I-在排序数组中查找数组I
subtitle:   LeetCode-Offer53-I-在排序数组中查找数组I #副标题
date:       2020-09-20            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

tag: easy，数组，二分查找

**题目：**

统计一个数字在排序数组中出现的次数。

**示例1：**

```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

**示例2：**

```
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

# 方法一：

**解题思路：**

>排序数组中的搜索问题，首先想到 二分法 解决。

排序数组nums中的所有数字target形成一个窗口，记窗口的左 / 右边界索引分别为 left 和 right ，分别对应窗口左边 / 右边的首个元素。

本题要求统计数字targe的出现次数，可转化为：使用二分法分别找到左边界left和右边界right ，易得数字target的数量为right - left - 1。

**算法解析：**

1. 初始化： 左边界 i = 0 ，右边界 j = len(nums) - 1 。
2. 循环二分： 当闭区间 [i, j][i,j] 无元素时跳出；
    1. 计算中点 m = (i + j) / 2（向下取整）；
    2. 若 nums[m] < target，则 target 在闭区间 [m + 1, j][m+1,j] 中，因此执行 i = m + 1；
    3. 若 nums[m] > target，则 target在闭区间 [i, m - 1][i,m−1] 中，因此执行 j = m - 1；
    4. 若 nums[m] = target，则右边界 right 在闭区间 [m+1, j][m+1,j] 中；左边界 left 在闭区间 [i, m-1][i,m−1] 中。因此分为以下两种情况：
        1.  若查找 右边界 right ，则执行 i = m + 1 ；（跳出时 i 指向右边界）
        2.  若查找 左边界 left ，则执行 j = m - 1 ；（跳出时 jj 指向左边界）
3. 返回值： 应用两次二分，分别查找 right 和 left ，最终返回 right - left - 1 即可。

**效率优化：**

>以下优化基于：查找完右边界 right = iright=i 后，则 nums[j]nums[j] 指向最右边的 targettarget （若存在）。

1.  查找完右边界后，可用 nums[j] = j判断数组中是否包含 target，若不包含则直接提前返回 0 ，无需后续查找左边界。
2.  查找完右边界后，左边界 left 一定在闭区间 [0, j][0,j] 中，因此直接从此区间开始二分查找即可。

##### 复杂度分析：

-   **时间复杂度 O(log N)：** 二分法为对数级别复杂度。
-   **空间复杂度 O(1)：** 几个变量使用常数大小的额外空间。

```python
class Solution:
    def search(self, nums: [int], target: int) -> int:
        # 搜索右边界 right
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] <= target: i = m + 1
            else: j = m - 1
        right = i
        # 若数组中无 target ，则提前返回
        if j >= 0 and nums[j] != target: return 0
        # 搜索左边界 left
        i = 0
        while i <= j:
            m = (i + j) // 2
            if nums[m] < target: i = m + 1
            else: j = m - 1
        left = j
        return right - left - 1
```

>执行用时：28 ms, 在所有 Python3 提交中击败了99.73%的用户
>
>内存消耗：14.2 MB, 在所有 Python3 提交中击败了68.42%的用户

[Link](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/solution/mian-shi-ti-53-i-zai-pai-xu-shu-zu-zhong-cha-zha-5/)