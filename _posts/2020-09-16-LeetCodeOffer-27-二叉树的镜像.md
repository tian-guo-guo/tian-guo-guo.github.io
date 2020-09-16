---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer27-二叉树的镜像
subtitle:   LeetCode-Offer27-二叉树的镜像 #副标题
date:       2020-09-16            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/) [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

tag: easy，二叉树，递归，辅助栈（队列）

**题目：**

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

 4
  /  \
 2   7
 / \  / \
1  3 6  9

镜像输出：

4
  /  \
 7   2
 / \  / \
9  6 3  1

**示例 1：**

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

# 方法一：递归法

>二叉树镜像定义： 对于二叉树中任意节点 root，设其左 / 右子节点分别为 left, right ；则在二叉树的镜像中的对应 root节点，其左 / 右子节点分别为 right, left 。

-   复杂度分析：
    -   时间复杂度 O(N) ： 其中 N 为二叉树的节点数量，建立二叉树镜像需要遍历树的所有节点，占用 O(N)时间。
    -   空间复杂度 O(N) ： 最差情况下（当二叉树退化为链表），递归时系统需使用 O(N)大小的栈空间。

-   解题思路：
    -   根据二叉树镜像的定义，考虑递归遍历（dfs）二叉树，交换每个节点的左 / 右子节点，即可生成二叉树的镜像。
    -   递归解析：
        1. 终止条件： 当节点 root为空时（即越过叶节点），则返回 null；
        2. 递推工作：
          1. 初始化节点 tmp，用于暂存 root的左子节点；
          2. 开启递归 右子节点 mirrorTree(root.right)，并将返回值作为 root的 左子节点 。
          3. 开启递归 左子节点 mirrorTree(tmp)，并将返回值作为 root的 右子节点 。
        3. 返回值： 返回当前节点 root；

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def mirrorTree(self, root):
        if not root:
            return
        root.left, root.right = self.mirrorTree(root.right), self.mirrorTree(root.left)
        return root
```

>执行用时：36 ms, 在所有 Python3 提交中击败了90.35%的用户
>
>内存消耗：13.4 MB, 在所有 Python3 提交中击败了27.56%的用户

# 方法二：辅助栈（队列）

>   利用栈（或队列）遍历树的所有节点 node，并交换每个 node的左 / 右子节点。

- 算法流程：
    1. 特例处理： 当 root为空时，直接返回 null；
    2. 初始化： 栈（或队列），本文用栈，并加入根节点 root。
    3. 循环交换： 当栈 stack为空时跳出；
        1. 出栈： 记为 node；
        2. 添加子节点： 将 node左和右子节点入栈；
        3. 交换： 交换 node的左 / 右子节点。
    4. 返回值： 返回根节点 root 。
- 复杂度分析：
    - 时间复杂度 O(N)： 其中 N为二叉树的节点数量，建立二叉树镜像需要遍历树的所有节点，占用 O(N)时间。
    - 空间复杂度 O(N)： 最差情况下（当为满二叉树时），栈 stack最多同时存储 N/2个节点，占用 O(N)额外空间。

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def mirrorTree(self, root):
        if not root:
            return
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            node.left, node.right = node.right, node.left
        return root
```

>执行用时：32 ms, 在所有 Python3 提交中击败了97.16%的用户
>
>内存消耗：13.4 MB, 在所有 Python3 提交中击败了33.78%的用户

[Link](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/solution/mian-shi-ti-27-er-cha-shu-de-jing-xiang-di-gui-fu-/)

