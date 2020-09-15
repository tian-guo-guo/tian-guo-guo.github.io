---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer18-删除链表的节点
subtitle:   LeetCode-Offer18-删除链表的节点 #副标题
date:       2020-09-15            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

tag: easy，链表，双指针

**题目：**

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

**示例 1：**

```
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```

**示例 2：**

```
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

# 方法一：

- 解题思路：

>   删除值为 val 的节点可分为两步：定位节点、修改引用。

1.  定位节点： 遍历链表，直到 head.val == val 时跳出，即可定位目标节点。
2.  修改引用： 设节点 cur 的前驱节点为 pre ，后继节点为 cur.next ；则执行 pre.next = cur.next ，即可实现删除 cur 节点。

- 算法流程：
1. 特例处理： 当应删除头节点 head 时，直接返回 head.next 即可。
2. 初始化： pre = head , cur = head.next 。
3. 定位节点： 当 cur 为空 或 cur 节点值等于 val 时跳出。
    1. 保存当前节点索引，即 pre = cur 。
    2. 遍历下一节点，即 cur = cur.next 。
4. 删除节点： 若 cur 指向某节点，则执行 pre.next = cur.next 。（若 cur 指向 nullnull ，代表链表中不包含值为 val 的节点。
    返回值： 返回链表头部节点 head 即可。

- 复杂度分析：
    - 时间复杂度 O(N)： N为链表长度，删除操作平均需循环 N/2次，最差 N次。
    - 空间复杂度 O(1) ： cur, pre 占用常数大小额外空间。

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteNode(self, head, val):
        if head.val == val:
            return head.next
        pre, cur = head, head.next
        while cur and cur.val != val:
            pre, cur = cur, cur.next
        if cur:
            pre.next = cur.next
        return head
```

>执行用时：44 ms, 在所有 Python3 提交中击败了87.05%的用户
>
>内存消耗：13.9 MB, 在所有 Python3 提交中击败了61.38%的用户

[Link](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/solution/mian-shi-ti-18-shan-chu-lian-biao-de-jie-dian-sh-2/)