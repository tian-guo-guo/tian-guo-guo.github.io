---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer09-用两个栈实现队列
subtitle:   LeetCode-Offer09-用两个栈实现队列 #副标题
date:       2020-09-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

tag: medium，递归，树

**题目：**

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

**示例1：**

```
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
```

**示例2：**

```
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```

# 方法一：双栈实现队列

思想：

用两个栈来实现。

比如栈A=[1,2,3]，栈B=[]，A出栈到B，B=[3,2,1]，然后B再出栈，就相当于删除了栈A的元素，实现了队列



- 栈无法实现队列功能： 栈底元素（对应队首元素）无法直接删除，需要将上方所有元素出栈。
- 双栈可实现列表倒序： 设有含三个元素的栈 A = [1,2,3]和空栈 B = []。若循环执行 A 元素出栈并添加入栈 B ，直到栈 A 为空，则 A = [] , B = [3,2,1] ，即 栈 B 元素实现栈 A 元素倒序 。
- 利用栈 B 删除队首元素： 倒序后，B 执行出栈则相当于删除了 A 的栈底元素，即对应队首元素。

我们可以设计栈 `A` 用于加入队尾操作，栈 `B` 用于将元素倒序，从而实现删除队首元素。

-   加入队尾 appendTail()函数： 将数字 val 加入栈 A 即可。
-   删除队首deleteHead()函数： 有以下三种情况。
    -   当栈 B 不为空： B中仍有已完成倒序的元素，因此直接返回 B 的栈顶元素。
    -   否则，当 A 为空： 即两个栈都为空，无元素，因此返回 -1−1 。
    -   否则： 将栈 A 元素全部转移至栈 B 中，实现元素倒序，并返回栈 B 的栈顶元素。

```python
class CQueue:
    def __init__(self):
        self.A, self.B = [], []
       
    def appendTail(self, value):
        self.A.append(value)
        
    def deleteHead(self):
        if self.B:
            return self.B.pop()
        if not self.A:
            return -1
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()
```

-   时间复杂度： appendTail()函数为 O(1) ；deleteHead() 函数在 N 次队首元素删除操作中总共需完成 N 个元素的倒序。
-   空间复杂度 O(N)： 最差情况下，栈 A 和 B 共保存 N 个元素。

>执行用时：580 ms, 在所有 Python3 提交中击败了55.50%的用户
>
>内存消耗：16.9 MB, 在所有 Python3 提交中击败了76.19%的用户

[Link](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/solution/mian-shi-ti-09-yong-liang-ge-zhan-shi-xian-dui-l-2/)

