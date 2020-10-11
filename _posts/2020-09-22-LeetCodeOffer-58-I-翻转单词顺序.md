---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer58-I-翻转单词顺序151翻转字符串里的单词
subtitle:   LeetCode-Offer58-I-翻转单词顺序151翻转字符串里的单词 #副标题
date:       2020-09-22            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/) [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

tag: easy，字符串，双指针

**题目：**

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

**示例1：**

```
输入: "the sky is blue"
输出: "blue is sky the"
```

**示例2：**

```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```

**示例3：**

```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

# 方法一：双指针

**算法解析：**

- 倒序遍历字符串 s ，记录单词左右索引边界 i , j ；
- 每确定一个单词的边界，则将其添加至单词列表 res ；
- 最终，将单词列表拼接为字符串，并返回即可。

**复杂度分析：**

-   时间复杂度 O(N) ： 其中 N 为字符串 s 的长度，线性遍历字符串。
-   空间复杂度 O(N) ： 新建的 list(Python) 或 StringBuilder(Java) 中的字符串总长度 ≤N ，占用 O(N) 大小的额外空间。

```python
class Solution:
    def reverseWords(self, s):
        s = s.strip() # 删除首尾空格
        i = j = len(s) - 1
        res = []
        while i >= 0:
            while i >= 0 and s[i] != ' ': 
                i -= 1 # 搜索首个空格
                
            res.append(s[i + 1: j + 1]) # 添加单词
            
            while s[i] == ' ': 
                i -= 1 # 跳过单词间空格
                
            j = i # j 指向下个单词的尾字符
            
        return ' '.join(res) # 拼接并返回
```

>执行用时：40 ms, 在所有 Python3 提交中击败了81.04%的用户
>
>内存消耗：13.6 MB, 在所有 Python3 提交中击败了25.25%的用户

# 方法二：分割 + 倒序

利用 “字符串分割”、“列表倒序” 的内置函数 *（面试时不建议使用）* ，可简便地实现本题的字符串翻转要求。

**算法解析：**

Python ： 由于 split() 方法将单词间的 “多个空格看作一个空格” （参考自 split()和split(' ')的区别 ），因此不会出现多余的 “空单词” 。因此，直接利用 reverse() 方法翻转单词列表 strs ，拼接为字符串并返回即可。

```python
class Solution:
    def reverseWords(self, s):
        s = s.strip() # 删除首尾空格
        
        strs = s.split() # 分割字符串
        
        strs.reverse() # 翻转单词列表
        
        return ' '.join(strs) # 拼接为字符串并返回
```

```python
class Solution:
    def reverseWords(self, s):
        return ' '.join(s.strip().split()[::-1])
```



[Link](https://leetcode-cn.com/problems/reverse-words-in-a-string/solution/151-fan-zhuan-zi-fu-chuan-li-de-dan-ci-shuang-zh-2/)

