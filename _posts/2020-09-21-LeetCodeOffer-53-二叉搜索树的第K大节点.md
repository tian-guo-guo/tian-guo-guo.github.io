---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer54-二叉搜索树的第k大节点
subtitle:   LeetCode-Offer54-二叉搜索树的第k大节点 #副标题
date:       2020-09-21            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

tag: easy，二叉树，二叉搜索树，递归

**题目：**

给定一棵二叉搜索树，请找出其中第k大的节点。

**示例1：**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

**示例2：**

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

# 方法一：

**解题思路：**

>   本文解法基于此性质：二叉搜索树的中序遍历为 递增序列 。

-   根据以上性质，易得二叉搜索树的 中序遍历倒序 为 递减序列 。
-   因此，求 “二叉搜索树第 k 大的节点” 可转化为求 “此树的中序遍历倒序的第 k 个节点”。

>   **中序遍历** 为 “左、根、右” 顺序，递归法代码如下：

```python
# 打印中序遍历
def dfs(root):
    if not root: return
    dfs(root.left)  # 左
    print(root.val) # 根
    dfs(root.right) # 右
```

>**中序遍历的倒序** 为 “右、根、左” 顺序，递归法代码如下：

```python
# 打印中序遍历倒序
def dfs(root):
    if not root: return
    dfs(root.right) # 右
    print(root.val) # 根
    dfs(root.left)  # 左
```

-   为求第 k 个节点，需要实现以下 三项工作 ：
    1.  递归遍历时计数，统计当前节点的序号；
    2.  递归到第 k 个节点时，应记录结果 res ；
    3.  记录结果后，后续的遍历即失去意义，应提前终止（即返回）。

**递归解析：**

1. 终止条件： 当节点 rootroot 为空（越过叶节点），则直接返回；
2. 递归右子树： 即 dfs(root.right)；
3. 三项工作：
    1. 提前返回： 若 k = 0k=0 ，代表已找到目标节点，无需继续遍历，因此直接返回；
    2. 统计序号： 执行 k=k−1 （即从 k 减至 0 ）；
    3. 记录结果： 若 k=0 ，代表当前节点为第 k 大的节点，因此记录 res=root.val ；
4. 递归左子树： 即 dfs(root.left) ；

- 复杂度分析：
    - 时间复杂度 O(N) ： 当树退化为链表时（全部为右子节点），无论 k 的值大小，递归深度都为 N ，占用 O(N) 时间。
    - 空间复杂度 O(N)O(N) ： 当树退化为链表时（全部为右子节点），系统使用 O(N)O(N) 大小的栈空间。

```python
class TreeNode:
    def __init__(self, x):
        self.left = None
        self.val = x
        self.right = None
        
class Solution:
    def kthLargest(self, root, k):
        def dfs(root):
            if not root: return
            dfs(root.right)
            if self.k == 0: return
            self.k -= 1
            if self.k == 0: self.res = root.val
            dfs(root.left)

        self.k = k
        dfs(root)
        return self.res
```

>执行用时：56 ms, 在所有 Python3 提交中击败了95.70%的用户
>
>内存消耗：17.4 MB, 在所有 Python3 提交中击败了51.78%的用户

[Link](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/solution/mian-shi-ti-54-er-cha-sou-suo-shu-de-di-k-da-jie-d/)

