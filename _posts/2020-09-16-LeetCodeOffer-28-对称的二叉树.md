---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer28-对称的二叉树
subtitle:   LeetCode-Offer28-对称的二叉树 #副标题
date:       2020-09-16            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

tag: easy，递归，二叉树

**题目：**

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

   1
  / \
 2  2
 / \ / \
3  4 4  3

但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

   1
  / \
 2  2
  \  \
  3   3

**示例 1：**

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 1：**

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

# 方法一：递归法

>二叉树镜像定义： 对于二叉树中任意节点 root，设其左 / 右子节点分别为 left, right ；则在二叉树的镜像中的对应 root节点，其左 / 右子节点分别为 right, left 。

- 解题思路：
    - 对称二叉树定义： 对于树中 任意两个对称节点 L 和 R ，一定有：
        - L.val = R.val：即此两对称节点值相等。
        - L.left.val = R.right.val：即 L 的 左子节点 和 R 的 右子节点 对称；
        - L.right.val = R.left.val：即 LL 的 右子节点 和 RR 的 左子节点 对称。
    - 根据以上规律，考虑从顶至底递归，判断每对节点是否对称，从而判断树是否为对称二叉树。

- 算法流程：
- isSymmetric(root) ：
    - 特例处理： 若根节点 root 为空，则直接返回 true 。
    - 返回值： 即 recur(root.left, root.right) ;
- recur(L, R) ：
- 终止条件：
    - 当 L 和 R 同时越过叶节点： 此树从顶至底的节点都对称，因此返回 true ；
    - 当 L 或 R 中只有一个越过叶节点： 此树不对称，因此返回 false ；
    - 当节点 L 值≠节点 R 值： 此树不对称，因此返回 false ；
- 递推工作：
    - 判断两节点 L.left 和 R.right 是否对称，即 recur(L.left, R.right) ；
    - 判断两节点 L.right和R.leftt 是否对称，即 recur(L.right, R.left) ；
- 返回值： 两对节点都对称时，才是对称树，因此用与逻辑符 && 连接。

-   复杂度分析：
    -   时间复杂度 O(N)： 其中 N 为二叉树的节点数量，每次执行 recur() 可以判断一对节点是否对称，因此最多调用 N/2次 recur() 方法。
    -   空间复杂度 O(N) ： 最差情况下（见下图），二叉树退化为链表，系统使用 O(N) 大小的栈空间。

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def isSymmetric(self, root):
        def recur(L, R):
            if not L and not R:
                return True
            if not L or not R or L.val != R.val:
                return False
            return recur(L.left, R.right) and recur(L.right, R.left)
        return recur(root.left, root.right) if root else True
```

>执行用时：40 ms, 在所有 Python3 提交中击败了90.28%的用户
>
>内存消耗：13.2 MB, 在所有 Python3 提交中击败了100.00%的用户

[Link](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/solution/mian-shi-ti-28-dui-cheng-de-er-cha-shu-di-gui-qing/)