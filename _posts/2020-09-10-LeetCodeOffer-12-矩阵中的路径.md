---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer12-矩阵中的路径
subtitle:   LeetCode-Offer12-矩阵中的路径 #副标题
date:       2020-09-10            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/) [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

tag: medium，深度优先搜索DFS

**题目：**

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","**b**","c","e"],
["s","**f**","**c**","s"],
["a","d","**e**","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。  

示例 1：

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

示例 2：

```
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

# 方法一：

>   本问题是典型的矩阵搜索问题，可使用 **深度优先搜索（DFS）+ 剪枝** 解决。

- 算法原理：
    - 深度优先搜索： 可以理解为暴力法遍历矩阵中所有字符串可能性。DFS 通过递归，先朝一个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。
    - 剪枝： 在搜索中，遇到 这条路不可能和目标字符串匹配成功 的情况（例如：此矩阵元素和目标字符不同、此元素已被访问），则应立即返回，称之为 可行性剪枝 。



这道题非常典型，可以排到LeetCode经典题型的前5名，好好学会。

<iframe width="560" height="315" src="https://www.youtube.com/embed/1zSg1WdmhIs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**思路：**

首先需要有一个DFS的算法，go over这个board，

其次需要一个起始点，才能四散的去搜索这个board，所以需要两个for loop，一个go through rows，一个go through coulums，当定位到起始点的时候，就用DFS的算法不停的去找到所有的available position。

```python
class Solution:
    def exist(self, board, word):
        for i in range(len(board)): # matrix的行数
            
            for j in range(len(board[0])): # matrix的列数
                
                if self.helper(board, i, j, word, 0): # helper函数去nevigate，需要传入board，i，j，这个词Word以及起始的坐标0
                
                	return True # 如果满足情况就return true
                
              # 这里也有可能找到满足情况的序列
        
        return False # 如果都找不到那就return False
    
    def helper(self, board, i, j, word, wordIndex):
        if wordIndex == len(word):
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[wordIndex] != board[i][j]:
            return False

        board[i][j] = "#"
        found = self.helper(board, i+1, j, word, wordIndex+1) \
	            or self.helper(board, i, j+1, word, wordIndex+1) \
                or self.helper(board, i, j-1, word, wordIndex+1) \
                or self.helper(board, i-1, j, word, wordIndex+1)
		
        board[i][j] = word[wordIndex]
		
        return found
```

>执行用时：256 ms, 在所有 Python3 提交中击败了43.58%的用户
>
>内存消耗：14.4 MB, 在所有 Python3 提交中击败了93.39%的用户

[Link](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/solution/mian-shi-ti-12-ju-zhen-zhong-de-lu-jing-shen-du-yo/)

