---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer30-包含min函数的栈155最小栈
subtitle:   LeetCode-Offer30-包含min函数的栈155最小栈 #副标题
date:       2020-09-17            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/) [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

tag: easy，栈

**题目：**

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

**示例 1：**

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

# 方法一：

-   解题思路：

    >   普通栈的 push() 和 pop() 函数的复杂度为 O(1) ；而获取栈最小值 min() 函数需要遍历整个栈，复杂度为 O(N) 。

-   本题难点： 将 min() 函数复杂度降为 O(1) ，可通过建立辅助栈实现；
    -   数据栈 A ： 栈 A 用于存储所有元素，保证入栈 push() 函数、出栈 pop() 函数、获取栈顶 top() 函数的正常逻辑。
    -   辅助栈 B ： 栈 B 中存储栈 A 中所有 非严格降序 的元素，则栈 A 中的最小元素始终对应栈 B 的栈顶元素，即 min() 函数只需返回栈 B 的栈顶元素即可。
-   因此，只需设法维护好 栈 B 的元素，使其保持非严格降序，即可实现 min() 函数的 O(1) 复杂度。

-   函数设计：
    -   push(x) 函数： 重点为保持栈 BB 的元素是 非严格降序 的。
        1.  将 x 压入栈 A （即 A.add(x) ）；
        2.  若 ① 栈 B 为空 或 ② x 小于等于 栈 B 的栈顶元素，则将 x 压入栈 B （即 B.add(x) ）。
    -   pop() 函数： 重点为保持栈 A, BA,B 的 元素一致性 。
        1. 执行栈 A 出栈（即 A.pop() ），将出栈元素记为 yy ；
        2. 若 yy 等于栈 BB 的栈顶元素，则执行栈 B 出栈（即 B.pop() ）。
    -   top() 函数： 直接返回栈 AA 的栈顶元素即可，即返回 A.peek() 。
    -   min() 函数： 直接返回栈 B 的栈顶元素即可，即返回 B.peek() 。

-   复杂度分析：
    -   时间复杂度 O(1)： push(), pop(), top(), min() 四个函数的时间复杂度均为常数级别。
    -   空间复杂度 O(N)： 当共有 N 个待入栈元素时，辅助栈 B 最差情况下存储 N 个元素，使用 O(N) 额外空间。

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A, self.B = [], []

    def push(self, x):
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self):
        if self.A.pop() == self.B[-1]:
            self.B.pop()

    def top(self):
        return self.A[-1]

    def min(self):
        return self.B[-1]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```

>执行用时：116 ms, 在所有 Python3 提交中击败了14.11%的用户
>
>内存消耗：16.5 MB, 在所有 Python3 提交中击败了61.80%的用户

[Link](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/solution/mian-shi-ti-30-bao-han-minhan-shu-de-zhan-fu-zhu-z/)

