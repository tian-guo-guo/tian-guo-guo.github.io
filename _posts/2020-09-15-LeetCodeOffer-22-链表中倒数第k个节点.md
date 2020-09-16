---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer22-链表中倒数第k个节点
subtitle:   LeetCode-Offer22-链表中倒数第k个节点 #副标题
date:       2020-09-15            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

tag: easy，链表，双指针

**题目：**

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

**示例 1：**

```
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
```

# 方法一：

-   解题思路：
    -   第一时间想到的解法：
        -   先遍历统计链表长度，记为 nn ；
        -   设置一个指针走 (n-k)(n−k) 步，即可找到链表倒数第 kk 个节点。
    -   使用双指针则可以不用统计链表长度。

- 算法流程：
    1. 初始化： 前指针 former 、后指针 latter ，双指针都指向头节点 head 。
    2. 构建双指针距离： 前指针 former 先向前走 k 步（结束后，双指针 former 和 latter 间相距 k 步）。
    3. 双指针共同移动： 循环中，双指针 former 和 latter 每轮都向前走一步，直至 former 走过链表 尾节点 时跳出（跳出后， latter 与尾节点距离为 k-1，即 latter 指向倒数第 k 个节点）。
    4. 返回值： 返回 latter 即可。
- 复杂度分析：
    - 时间复杂度 O(N) ： N 为链表长度；总体看， former 走了 N 步， latter 走了 (N−k) 步。
    - 空间复杂度 O(1) ： 双指针 former , latter 使用常数大小的额外空间。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = Node
        
class Solution:
    def getKthFromEnd(self, head, k):
        former, latter = head, head
        for _ in range(k):
            former = former.next
        while former:
            former, latter = former.next, latter.next
        return latter
```

>执行用时：28 ms, 在所有 Python3 提交中击败了99.65%的用户
>
>内存消耗：13.6 MB, 在所有 Python3 提交中击败了73.64%的用户

[Link](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/solution/mian-shi-ti-22-lian-biao-zhong-dao-shu-di-kge-j-11/)