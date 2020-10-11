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

# [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/) [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

tag: easy，二叉树，递归

**题目：**

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

>   一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过1。

**示例1：**

给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回 `true` 。

**示例2：**

给定二叉树 `[1,2,2,3,3,null,null,4,4]`，

```
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
```

返回 `false` 。

# 方法一：从顶至底（暴力法）

>此方法容易想到，但会产生大量重复计算，时间复杂度较高。

思路是构造一个获取当前节点最大深度的方法 depth(root) ，通过比较此子树的左右子树的最大高度差abs(depth(root.left) - depth(root.right))，来判断此子树是否是二叉平衡树。若树的所有子树都平衡时，此树才平衡。

**算法流程：**

-   isBalanced(root) ：判断树 root 是否平衡
    -   特例处理： 若树根节点 root 为空，则直接返回 true ；
    -   返回值： 所有子树都需要满足平衡树性质，因此以下三者使用与逻辑 \&\&&& 连接；
        1. abs(self.depth(root.left) - self.depth(root.right)) <= 1 ：判断 当前子树 是否是平衡树；
        2. self.isBalanced(root.left) ： 先序遍历递归，判断 当前子树的左子树 是否是平衡树；
        3. self.isBalanced(root.right) ： 先序遍历递归，判断 当前子树的右子树 是否是平衡树；
-   depth(root) ： 计算树 root 的最大高度
  -   终止条件： 当 root 为空，即越过叶子节点，则返回高度 00 ；
  -   返回值： 返回左 / 右子树的最大高度加 11 。

**复杂度分析：**

-   时间复杂度 O(Nlog_2 N)： 最差情况下， isBalanced(root) 遍历树所有节点，占用 O(N) ；判断每个节点的最大高度 depth(root) 需要遍历 各子树的所有节点 ，子树的节点数的复杂度为 O(log_2 N)。
-   空间复杂度 O(N)： 最差情况下（树退化为链表时），系统递归需要使用 O(N) 的栈空间。

```python
class TreeNode:
    def __init__(self, x):
        self.left = None
        self.val = x
        self.right = None
        
class Solution:
    def isBalanced(self, root):
        if not root: return True
        return abs(self.depth(root.left) - self.depth(root.right)) <= 1 and \
            self.isBalanced(root.left) and self.isBalanced(root.right)

    def depth(self, root):
        if not root: return 0
        return max(self.depth(root.left), self.depth(root.right)) + 1
```

>执行用时：72 ms, 在所有 Python3 提交中击败了51.20%的用户
>
>内存消耗：16.9 MB, 在所有 Python3 提交中击败了93.75%的用户

[Link](https://leetcode-cn.com/problems/balanced-binary-tree/solution/balanced-binary-tree-di-gui-fang-fa-by-jin40789108/)