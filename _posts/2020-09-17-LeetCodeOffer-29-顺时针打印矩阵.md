---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer29-顺时针打印矩阵54螺旋矩阵
subtitle:   LeetCode-Offer29-顺时针打印矩阵54螺旋矩阵 #副标题
date:       2020-09-17            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/) [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

难度中等491

tag: easy，矩阵

**题目：**

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

**示例 1：**

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 1：**

```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

# 方法一：

-   解题思路：

    >   根据题目示例 matrix = [[1,2,3],[4,5,6],[7,8,9]] 的对应输出 [1,2,3,6,9,8,7,4,5] 可以发现，顺时针打印矩阵的顺序是 “从左向右、从上向下、从右向左、从下向上” 循环。

- 因此，考虑设定矩阵的“左、上、右、下”四个边界，模拟以上矩阵遍历顺序。

- 算法流程：

    1. 空值处理： 当 matrix 为空时，直接返回空列表 [] 即可。
    2. 初始化： 矩阵 左、右、上、下 四个边界 l , r , t , b ，用于打印的结果列表 res 。
    3. 循环打印： “从左向右、从上向下、从右向左、从下向上” 四个方向循环，每个方向打印中做以下三件事 （各方向的具体信息见下表） ；
        1. 根据边界打印，即将元素按顺序添加至列表 res 尾部；
        2. 边界向内收缩 11 （代表已被打印）；
        3. 判断是否打印完毕（边界是否相遇），若打印完毕则跳出。
    4. 返回值： 返回 res 即可。

    打印方向	1. 根据边界打印	2. 边界向内收缩	3. 是否打印完毕
    从左向右	左边界l ，右边界 r	上边界 t 加 11	是否 t > b
    从上向下	上边界 t ，下边界b	右边界 r 减 11	是否 l > r
    从右向左	右边界 r ，左边界l	下边界 b 减 11	是否 t > b
    从下向上	下边界 b ，上边界t	左边界 l 加 11	是否 l > r

- 复杂度分析：

    - 时间复杂度 O(MN) ： M, N分别为矩阵行数和列数。
    - 空间复杂度 O(1)： 四个边界 l , r , t , b 使用常数大小的 额外 空间（ res 为必须使用的空间）。

```python
class Solution:
    def spiralOrder(self, matrix):
        if not matrix:
            return []
        l, r, t, b, res = 0, len(matrix[0])-1, 0, len(matrix)-1, []
        while True:
            for i in range(l, r + 1): res.append(matrix[t][i]) # left to right
                
            t += 1
            if t > b: break
            for i in range(t, b + 1): res.append(matrix[i][r]) # top to bottom
                
            r -= 1
            if l > r: break
            for i in range(r, l - 1, -1): res.append(matrix[b][i]) # right to left
                
            b -= 1
            if t > b: break
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # bottom to top
                
            l += 1
            if l > r: break
        return res
```

>执行用时：40 ms, 在所有 Python3 提交中击败了95.78%的用户
>
>内存消耗：13.6 MB, 在所有 Python3 提交中击败了28.01%的用户

[Link](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/solution/mian-shi-ti-29-shun-shi-zhen-da-yin-ju-zhen-she-di/)