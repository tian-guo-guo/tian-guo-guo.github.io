---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer24-反转链表 206反转链表
subtitle:   LeetCode-Offer24-反转链表 206反转链表 #副标题
date:       2020-09-15            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/) [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

tag: easy，链表，双指针

**题目：**

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

**示例 1：**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

# 方法一：

-   解题思路：
    -   相当于定义两个指针，last和head。 head.next= last表示head的下一个指向last。剩下两个等号相当于last和head两个指针同时后移。
    

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = Node
        
class Solution:
    def reverseList(self, head):
        last = None
        while head:
            head.next, last, head  = last, head, head.next
        return last
```

>执行用时：52 ms, 在所有 Python3 提交中击败了29.86%的用户
>
>内存消耗：14.4 MB, 在所有 Python3 提交中击败了98.45%的用户

[Link](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/solution/6xing-dai-ma-yi-ci-bian-li-by-chitoseyono/)



# 方法二：视频讲解

<iframe width="560" height="315" src="https://www.youtube.com/embed/C6LzmH20GNk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

-   解题思路

    > 新建一个叫dummy的linklist，倒着一个一个值放进去。

    可能需要2-3个指针，起始点设置为dummy，head.val放到dummy.next里去，是想reverse的时候扩展出来一个新的linklist， 也就是dummy的第一个值也就是head的第一个值

    ```
    # create dummy, dummy.next = None
    # starting from node.val = 1
    # dummy.next, head.next, head = head, dummy.next, head.next
    # dummy -> 1 -> NULL
    # iteration head = 2, dummy.next = 1
    ```

    ```python
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None
            
    class Solution:
        def reverseList(self, head):
            dummy = ListNode(float('-inf')) # 一般来说新建一个值都是float('-inf')
            
            while head:
                dummy.next, head.next, head = head, dummy.next, head.next
            return dummy.next
    ```

    >执行用时：36 ms, 在所有 Python3 提交中击败了97.72%的用户
    >
    >内存消耗：14.5 MB, 在所有 Python3 提交中击败了82.70%的用户