---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer55-I-二叉树的深度104二叉树的最大深度
subtitle:   LeetCode-Offer55-I-二叉树的深度104二叉树的最大深度 #副标题
date:       2020-09-21            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/) [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

tag: easy，二叉树，递归，DFS，BFS

**题目：**

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

**示例1：**

给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

# 方法一：递归+深度优先搜索

1.  递归停止条件：
    -   如果`root`是`None`：返回`False`
2.  返回：
    -   返回左边的最大深度和右边的最大深度的最大值`+1`

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        return 0 if not root else max(self.maxDepth(root.left), self.maxDepth(root.right))+1
```

>执行用时：56 ms, 在所有 Python3 提交中击败了55.22%的用户
>
>内存消耗：15.2 MB, 在所有 Python3 提交中击败了35.95%的用户

# 方法二：广度优先搜索

本思路就是得到二叉树的[层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)，并取得长度。

每一次都保存上一次的节点，每循环一次，都给计数器增加`1`，每次都刷新为左和右（如果有的情况下），直到全部没有。

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        tmp, ret = [root], 1
        while tmp:
            ret, tmp = ret+1, sum([([i.left] if i.left else [])+([i.right] if i.right else []) for i in tmp], [])
        return ret-1
```



[Link](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/solution/python3-xiang-xi-di-gui-dfsbfs-by-ting-ting-28/)