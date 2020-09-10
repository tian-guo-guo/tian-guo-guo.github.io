---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer06-从头到尾打印链表
subtitle:   LeetCode-Offer06-从头到尾打印链表 #副标题
date:       2020-09-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

tag: easy，递归，栈，链表

**题目：**

**输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。**

**示例1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

# 方法一：递归法

利用递归，先走至链表末端，回溯时依次将节点值加入列表 ，这样就可以实现链表值的倒序输出。

-   递推阶段： 每次传入 head.next ，以 head == None（即走过链表尾部节点）为递归终止条件，此时返回空列表 [] 。
-   回溯阶段： 利用 Python 语言特性，递归回溯时每次返回 当前 list + 当前节点值 [head.val] ，即可实现节点的倒序输出。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def reversePrint(self, head):
        return self.reversePrint(head.next) + [head.val] if head else []
```

-   **时间复杂度 O(N)：** 遍历链表，递归 N*N* 次。
-   **空间复杂度 O(N)：** 系统递归需要使用 O(N)的栈空间。

>执行用时：140 ms, 在所有 Python3 提交中击败了9.77%的用户
>
>内存消耗：22.9 MB, 在所有 Python3 提交中击败了8.29%的用户

# 方法二：辅助栈法

-   入栈： 遍历链表，将各节点值 push 入栈。（Python 使用 append() 方法，Java借助 LinkedList 的addLast()方法）。

-   出栈： 将各节点值 pop 出栈，存储于数组并返回。（Python 直接返回 stack 的倒序列表，Java 新建一个数组，通过 popLast() 方法将各元素存入数组，实现倒序输出）。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = Node
        
class Solution:
    def reversePrint(self, head):
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]
```

- 时间复杂度 O(N)O(N)： 入栈和出栈共使用 O(N)O(N) 时间。

-   空间复杂度 O(N)： 辅助栈 stack 和数组 res 共使用 O(N) 的额外空间。

>执行用时：40 ms, 在所有 Python3 提交中击败了95.51%的用户
>
>内存消耗：15.4 MB, 在所有 Python3 提交中击败了27.68%的用户

[Link](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/mian-shi-ti-06-cong-wei-dao-tou-da-yin-lian-biao-d/)

