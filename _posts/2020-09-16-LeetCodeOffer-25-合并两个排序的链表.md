---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer25-合并两个排序的链表
subtitle:   LeetCode-Offer25-合并两个排序的链表 #副标题
date:       2020-09-16            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/) [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/) [果果](https://tianguoguo.fun/2020/07/11/LeetCode-21-%E5%90%88%E5%B9%B6%E4%B8%A4%E4%B8%AA%E6%9C%89%E5%BA%8F%E9%93%BE%E8%A1%A8/)

tag: easy，链表，双指针

**题目：**

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

**示例 1：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

# 方法一：

-   解题思路：
    -   根据题目描述， 链表 l_1, l_2是递增的，因此容易想到使用双指针 l_1和 l_2遍历两链表，根据 l_1.val和 l_2.val的大小关系确定节点添加顺序，两节点指针交替前进，直至遍历完毕。
    -   引入伪头节点： 由于初始状态合并链表中无节点，因此循环第一轮时无法将节点添加到合并链表中。解决方案：初始化一个辅助节点 dum作为合并链表的伪头节点，将各节点添加至 dum之后。

- 算法流程：
    1.  初始化： 伪头节点 dum，节点 cur指向 dum。
    2.  循环合并： 当 l_1或 l_2为空时跳出；
        1.  当 l_1.val < l_2.val时： cur的后继节点指定为 l_1，并 l_1向前走一步；
        2.  当 l_1.val≥l 2 .val 时： curcur 的后继节点指定为 l_2，并 l_2向前走一步 ；
        3.  节点 cur向前走一步，即 cur = cur.next。
    3.  合并剩余尾部： 跳出时有两种情况，即 l_1为空 或 l_2为空。
        若 l_1≠null ： 将 l_1 添加至节点 cur之后；
        否则： 将 l_2添加至节点 cur之后。
    4.  返回值： 合并链表在伪头节点 dum之后，因此返回 dum.next即可。

- 复杂度分析：
    - 时间复杂度 O(M+N)： M, N分别为链表 l_1, l_2的长度，合并操作需遍历两链表。
    - 空间复杂度 O(1)： 节点引用 dum , cur使用常数大小的额外空间。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeTwoLists(self, l1, l2):
        cur = dum = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next, l1 = l1, l1.next
            else:
                cur.next, l2 = l2, l2.next
            cur = cur.next
        cur.next = l1 if l1 else l2
        return dum.next
```

>执行用时：64 ms, 在所有 Python3 提交中击败了69.74%的用户
>
>内存消耗：14.1 MB, 在所有 Python3 提交中击败了31.44%的用户