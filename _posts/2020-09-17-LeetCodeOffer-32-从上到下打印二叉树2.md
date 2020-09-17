---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer32-II从上到下打印二叉树II102二叉树的层序遍历
subtitle:   LeetCode-Offer32-II从上到下打印二叉树II102二叉树的层序遍历 #副标题
date:       2020-09-17            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/) [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

tag: easy，二叉树，广度优先搜索，队列

**题目：**

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3 
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```

# 方法一：

#### 解题思路：

I. 按层打印： 题目要求的二叉树的 从上至下 打印（即按层打印），又称为二叉树的 广度优先搜索（BFS）。BFS 通常借助 队列 的先入先出特性来实现。

II. 每层打印到一行： 将本层全部节点打印到一行，并将下一层全部节点加入队列，以此类推，即可分为多行打印。

-   算法流程：
    1. 特例处理： 当根节点为空，则返回空列表 [] ；
    2. 初始化： 打印结果列表 res = [] ，包含根节点的队列 queue = [root] ；
    3. BFS 循环： 当队列 queue 为空时跳出；
        1. 新建一个临时列表 tmp ，用于存储当前层打印结果；
        2. 当前层打印循环： 循环次数为当前层节点数（即队列 queue 长度）；
            1. 出队： 队首元素出队，记为 node；
            2. 打印： 将 node.val 添加至 tmp 尾部；
            3. 添加子节点： 若 node 的左（右）子节点不为空，则将左（右）子节点加入队列 queue ；
        3. 将当前层结果 tmp 添加入 res 。
    4. 返回值： 返回打印结果列表 res 即可。

-   复杂度分析：
    -   时间复杂度 O(N) ： N 为二叉树的节点数量，即 BFS 需循环 N 次。
    -   空间复杂度 O(N) ： 最差情况下，即当树为平衡二叉树时，最多有 N/2 个树节点同时在 queue 中，使用 O(N) 大小的额外空间。

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            tmp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return res
```

>执行用时：44 ms, 在所有 Python3 提交中击败了63.13%的用户
>
>内存消耗：13.7 MB, 在所有 Python3 提交中击败了37.81%的用户

[Link](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/solution/mian-shi-ti-32-ii-cong-shang-dao-xia-da-yin-er-c-5/)