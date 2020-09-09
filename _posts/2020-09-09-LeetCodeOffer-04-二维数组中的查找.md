---
layout:     post           # 使用的布局（不需要改）
title:      Offer04二维数组中的查找
subtitle:   Offer04二维数组中的查找 #副标题
date:       2020-09-09            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

tag: easy，数组

**题目：**

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**示例1：**

现有矩阵 matrix 如下：

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

# 方法一：

定位左下角，按照行列索引确定matrix中的数字，

起始行列i，j行列坐标分别为左下角的len(matirx)-1和0，

然后开始循环，行索引i要逐渐的减小，直到i>=0，列索引要逐渐的增大，直到j<len(matix[0])，

判断matrix中的数字[i] [j]与要找的数字大小的关系，进行行列索引的++--，找到了返回True，没有找到，出了循环返回False。

```python
class Solution:
    def findNumberIn2DArray(self, matrix, target):
        i, j = len(matrix) - 1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
            else:
                return True
        return False
```

-   时间复杂度 O(M+N) ：其中，N 和 M 分别为矩阵行数和列数，此算法最多循环 M+N次。
-   空间复杂度 O(1) : i, j 指针使用常数大小额外空间。

>执行用时：40 ms, 在所有 Python3 提交中击败了94.33%的用户
>
>内存消耗：17.8 MB, 在所有 Python3 提交中击败了52.12%的用户

[Link](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/solution/mian-shi-ti-04-er-wei-shu-zu-zhong-de-cha-zhao-zuo/)

