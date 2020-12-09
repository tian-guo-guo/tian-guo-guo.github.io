---
layout:     post           # 使用的布局（不需要改）
title:      LeeCode-write
subtitle:   LeeCode-write #副标题
date:       2020-12-09            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# LeeCode-write

# 数字

## LeetCode-07-整数反转-个位取值

**题目：**
 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
 输入: 123 输出: 321 输入: -123  输出: -321 输入: 120  输出: 21

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−2^31, 2^31 − 1]。[-214783648, 214783647]，如果反转后整数溢出那么就返回 0。
**思路：**

先不考虑符号，用abs，然后while loop这个数，个位temp：通过取余获得，数值num：存到num里，num每一轮都扩大10倍，左移一位，然后把temp加到最右边，余下的数a：因为已经把个位数移掉了，只需要保留10位数和10位数以上的数，a=int(a/10)。然后下一步判断输入是正还是负，以及索引越界的问题。

```python
def reverseStr(num):
    res = 0
    a = abs(num)
    while (a!=0):
        temp = a % 10
        res = res * 10 + temp
        a =int(a // 10)
    if num > 0 and res < 2147483647:
        return res
    elif num < 0 and res >= -214783648:
        return -res
    else: 
        return 0
print(reverseStr(-123))
```

收获：

>返回一个结果，那先定义一个res=0
>
>先不判断一个数的正负，用abs(num)
>
>判断一个数是不是等于0，可以while(n)或者while(n!=0)
>
>取一个数的个位，可以与10取余，temp = num % 10
>
>一个数每回扩大10倍，可以乘以10再加上temp
>
>一个数去掉最后一位，可以除以10并向下取整int(num // 10)
>
>最后结果判断，有一个索引越界的问题 

## LeetCode-09-回文数

**题目：**

判断一个整数是否是回文数。 输入: 121  输出: true  输入: -121  输出: false  输入: 10  输出: false

**方法一：一位一位的取**

如果一个数是负数，那一定不是回文数。

先取绝对值，然后个位temp取余，数值num乘10相加，剩下的数a，除以10取整。

然后判断翻转过来的数和原来的数是不是一样的，联合和负数就一起判断了。

```python
def isPalindrome(x):
    res = 0
    num = abs(x)
    while num:
        temp = num % 10
        res = res * 10 + temp
        num = int(num // 10)
    if x >= 0 and res == x:
        return True
    else:
        return False

print(isPalindrome(121))
```

**收获：**

>判断条件可以正确的和大于零的数一起判断。



**方法二：转化为字符串**

```python
def isPalindrome(x):
    s = str(x)
    return s[::-1] == str(x)

print(isPalindrome(-123))
```

**收获：**

>   转化为list之后直接反转比较



# 字符串

## 出逃密码-list(map)

**题目描述:**
在一次密室逃脱中，玩家小玺需要通过对手给定的N组英文语句来推出密码，只有这样才能逃出，否则将会受到惩罚。N组英文语句中的单词分别用空格分隔，这里的单词是连续的、不是空格的单词。小玺需要输出每个语句中的单词数量来推出密码。请你编写程序帮助小玺快速找到出逃的密码。
**输入描述:**
输入第一行为英文语句的数量N
接下来每行为英文语句。
**输出描述:**
输出各英文语句中的单词数量，不同语句的数量用空格分隔。
**样例输入:**
2
"Hello my name is Xiaoxi"
"I can do it"
**输出:**
5 4

```python
n = int(input().strip())
res = []
for i in range(n):
    res.append(str(len(list(map(str, input().strip().split())))))
print(' '.join(res))
```

**收获：**

>   学到了list(map(function, str))的用法

## LeetCode-[14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

**题目：**

编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串 `""`。

输入: ["flower","flow","flight"]  输出: "fl"

输入: ["dog","racecar","car"]  输出: ""

**思路一：**

一个一个字符的比较。先取出第一个词strs[0]，然后从strs[1:]第2个词开始一一比较，相等的话比较下一个，不相等直接返回strs[0][i]

```python
def longgestCommonPrefix(strs):
    if not strs:
        return ''
    for i in range(len(strs[0])):
        for string in strs[1:]:
            if i > len(string) or string[i] != strs[0][i]:
                return strs[0][:i]
    return strs[0]

print(longgestCommonPrefix(["flower", "flow", "flight"]))
```

**收获：**

这道题相当于二维list了，一维是所有的单词，二维是比较所有单词的字母。

一维取第一个单词的长度来当索引下标了，因为要返回的是最长公共前缀。

二维就比较，得从第二个单词开始比较，当前单词的字母string[i]和第一个单词的字母strs[0] [i]是不是一样，

当出现不一样的时候就可以返回第一个单词的前几个字符strs[0] [:i]

遍历完了，都一样，那就是都是一样的前缀，就返回第一个单词。

**思路二：**

遍历strs里的每一个string，用set存储，看string[i]有几个值，只要一个的时候才加到result里，result+=sets.pop()，长度大于一个直接break，返回结果。

```python
def longestCommonPrefix(strs):
    result = ''
    i = 0
    while True:
        try:
            sets = set(string[i] for string in strs)
            if len(sets) == 1:
                result += sets.pop()
                i += 1
            else:
                break
        except Exception as e:
            break
    return result

print(longestCommonPrefix(["flower", "flow", "flight"]))
```

**收获：**

集合的方法！

用set的列表生成式遍历每一个位置的字符set(string[i] for string in strs)，看长度是不是1

如果长度是1，那就把set集合里的字符pop()出来，加到res里！

否则的话就break退出，别忘了i+=1！

另外，因为写了个while True死循环，所以可以用try except来break跳出！



# 字典

## 递归找到所有id name-递归

**题目描述：**

输入一个字典，key有id，name，children，children是一个列表里面可能会有0个到多个的id，name，children字典
**例子**
{‘id’: 123, ’name’: ”123, ’children’: [
    {‘id’: 123, ’name’: ”123, ’children’: [
        {‘id’: 123, ’name’: ”123, ’children’: []},
        {‘id’: 123, ’name’: ”123, ’children’: []},
        {‘id’: 123, ’name’: ”123, ’children’: []},
        {‘id’: 123, ’name’: ”123, ’children’: []}
    ]},
    {‘id’: 123, ’name’: ”123, ’children’: []},
    {‘id’: 123, ’name’: ”123, ’children’: []},
    {‘id’: 123, ’name’: ”123, ’children’: []},
    {‘id’: 123, ’name’: ”123, ’children’: []}
]}
**输出：**
找到字典中的所有id name对
返回[(id, name), (id, name), (id, name), (id, name), (id, name)]

```python
input_dic = {'id': 123, 'name': '123',
         'children': [{'id': 123, 'name': '123',
                       'children': [{'id': 123, 'name': '123', 'children': []},
                                    {'id': 123, 'name': '123', 'children': []},
                                    {'id': 123, 'name': '123', 'children': []},
                                    {'id': 123, 'name': '123', 'children': []}]},
                      {'id': 123, 'name': '123', 'children': []},
                      {'id': 123, 'name': '123', 'children': []},
                      {'id': 123, 'name': '123', 'children': []},
                      {'id': 123, 'name': '123', 'children': []}
                      ]}

class Solution:
    def getResult(self, childrenList):
        length = len(childrenList)
        if length == 0:
            return
        else:
            for item in childrenList:
                self.resultList.append((item['id'],item['name']))
                self.getResult(item['children'])
    def main(self, input_dic):
        self.resultList = []
        self.getResult([input_dic])
        return self.resultList

if __name__ == "__main__":
    s = Solution()
    print(s.main(input_dic))
    print(len(s.main(input_dic)))
```

**收获：**

>学到了递归的写法和字典的取值

## LeetCode-[13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

题目：

给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。numral_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

罗马数字 2 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做 XXVII, 即为 XX + V + II 。

罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。 X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。 

```
输入: "III"   输出: 3   输入: "IV"   输出: 4   输入: "IX"    输出: 9   输入: "LVIII"   输出: 58   解释: L = 50, V= 5, III = 3.   输入: "MCMXCIV"   输出: 1994   解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

**思路：**

“MCMXCIV”，如果后面的比前面的大，就认为是一对，值就是后面的减前面的，比如CM=1000-10，XC=100-10，要不然就是单个的。

最好的方法就是建立一个dictionary，把每一个symbol对应的value写到dictionary里面，

```python
def roman2Int(s):
    numral_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    res = 0 
    for i in range(len(s)):
        if i>0 and numral_map[s[i]] > numral_map[s[i-1]]:
            res += numral_map[s[i]] - 2 * numral_map[s[i-1]]
        else:
            res += numral_map[s[i]]
    return res

print(roman2Int('IX'))
```

**收获：**

>   根据下标给字符串里的每个字符绑定索引，然后再根据字典找到每个字符对应的数值！numral_map[s[i]]
>
>   因为涉及到当前数与前面的一个数，所以索引下标得从if i>0开始找起！
>
>   然后就是公式的判断了，因为前边res已经加过了numral_map[s[i]]，所以当找到后边的数比前边的数大的时候就要res += numral_map[s[i]] - 2 * numral_map[s[i-1]]



## LeetCode-20 有效的括号-dic[stack.pop()]

**题目描述：**

输入: "()"  输出: true  输入: "()[]{}"  输出: true  输入: "(]"  输出: false  输入: "([)]"  输出: false  输入: "{[]}"  输出: true

**思路一：**

用栈，先入后出的特点，遍历括号，即若遇到左括号入栈，遇到右括号时将对应栈顶左括号出栈，则遍历完所有括号后 stack 仍然为空； 

**注意：**

1. 建立哈希表 dic 构建左右括号对应关系：key 左括号，value 右括号；这样查询 2 个括号是否对应只需 O(1) 时间复杂度；

2. 栈 stack 为空： 此时 stack.pop() 操作会报错；因此，我们采用一个取巧方法，给 stack 赋初值 ? ，并在哈希表 dic 中建立 key:'?'value:'?'key: ′?′，value: ′? ′的对应关系予以配合。

3. 此时当 stack 为空且 c 为右括号时，可以正常提前返回 false；

4. 字符串 s 以左括号结尾： 此情况下可以正常遍历完整个 s，但 stack 中遗留未出栈的左括号；因此，最后需返回 len(stack) == 1，以判断是否是有效的括号组合。

时间复杂度 O(N)：正确的括号组合需要遍历 1 遍 s； 

空间复杂度 O(N)：哈希表和栈使用线性的空间大小。

```python
def isValid(s):
    stack = []
    dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
    for c in s:
        if c in dic:
            stack.append(c)
        elif dic[stack.pop()] != c:
            return False
    return len(stack) == 0

print(isValid("()(]{}"))
```

**收获：**

>学到了字典弹出一个数据是dic[strack.pop()]



# 列表

## 期末快速阅卷-异或^ eval

**题目描述:**
每到期末考完试，同学们就会迎来寒/暑假:但者师们却忙得不可开交，大量的阅卷工作让者师们身心俱疲。现在老师们想到了一个快速阅卷的方法:先给定两个答案(一个是正确答案，另一个是学生试卷上的答案)，两个答案的顺序是一对应的。请你编写-个程序， 帮助者师们快速检验两个答案是否一致。若答案一致输出itrue, 否则输出false.
**输入描述：**
第一行输入者师给定的正确答案;
第二行输入学生的答案。
**输出描述：**
若答案完全致输出true, 否则输出false.
**样例输入：**
[1,2,3]
[1,2,3]
**样例输出：**
true
**知识点：**
eval() 函数用来执行一个字符串表达式，并返回表达式的值。

>x = 7
>eval( '3 * x' )
>21
>eval('pow(2,2)')
>4
>eval('2 + 2')
>4
>n=81
>eval("n + 4")
>85

```python
answer = eval(input().strip())
student = eval(input().strip())
for i,j in zip(answer, student):
if i ^ j != 0:
print('false')
exit()
print('true')
```

**收获：**

>   学到了eval()的用法
>
>   以及判断两个数不一样用i^j != 0

## 优先理发-sorted, lambda

**题目描述:**
某大学的志愿者团队每个月部会去养老院为老人们理发。在毒理发老人的名单中，志愿吉得知有n位老人需要理发。根据惯例，应优先女性理发，且优先年龄最大的老人理发(不管年龄多大，只要是女性就会比男性优先理发)。请你根据名单中老人的性别和年龄，为志愿者排出最科学的理发顺序。
**输入描述：**
第一行为需理发老人的编号;
第二行为老人的性别，女性=1,男性=O;
第三行为老人的年龄n (ns1s100)↓用空格分隔:需要注意者人的性别和年龄按数字顺序是对应的
**输出描述：**
输出理发的顺序(即需理发老人的编号，且每个老人的编号用空档隔开)
**样例输入：**
1 2 3 4 5
1 0 1 0 1
65 67 87 76 98
**样例输出：**
5 3 1 4 2

```python
orders = [1, 2, 3, 4, 5]
genders = [1, 0, 1, 0, 1]
ages = [65, 67, 87, 76, 98]
# orders = list(map(int,input().split()))
# genders = list(map(int,input().split()))
# ages = list(map(int,input().split()))
womens = []
mans = []
for order, gender, age in zip(orders, genders, ages):
    if gender == 1:
        womens.append([order, age])
    else:
        mans.append([order, age])
result = []
for item in sorted(womens, key=lambda x: x[1], reverse=True):
    result.append(str(item[0]))
for item in sorted(mans, key=lambda x: x[1], reverse=True):
    result.append(str(item[0]))
print(" ".join(result))
```

**收获：**

>学到了通过zip给数据分类为women和men
>
>还有sorted排序返回一个新list，传list和其他关键字
>
>以及lambda args:表达式的用法

## LeetCode-26-删除排序数组中的重复项-原地修改数组

**题目：**

给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。不要使用额外的数组空间，原地修改，O(1) 的复杂度。nums = [1,1,2]，函数应该返回 2, 原数组为 1, 2。 nums = [0,0,1,1,1,2,2,3,3,4], 函数应该返回5, 并且原数组为 0, 1, 2, 3, 4。

**方法一：**

如果nums没有数据直接返回0，

否则的话设计一个记录有多少个不同数据的count值=0，

然后循环nums里的每一个数据，len(nums)

li[count]数组是从前往后有序排列的第1 2 3 4 5个数，li[i]是真实的那个数

找到实际li[i]的数与第一个数li[count]不相等的时候，然后把值赋给li[count]，count的数值就要加1，

最后返回count+1，count不能是1，因为假如nums=[1]的话，nums[count]就会报错，out of range

```python
def deldup(li):
    if not li:
        return li
    count = 0
    for i in range(len(li)):
        if li[count] != li[i]:
            count += 1
            li[count] = li[i]
    return count + 1

print(deldup([0,0,1,1,1,2,2,3,3,4]))
```

**收获：**

>学到了用count=0记录一个新list，逐个1 2 3 4 5，在原数组上直接操作数据

## LeetCode-27-移除元素

**题目：**

给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。不使用额外数组空间，O(1)原地修改输入数组。
 nums = [3,2,2,3], val = 3，函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
 nums = [0,1,2,2,3,0,4,2], val = 2，函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

**思路：**

双指针，一个指在array的开始，一个指在array的结束，当i<=last的时候，进行换。

如果当前值nums[i]==给定的val值，那么当前值跟最后一个值换一下，怎么删除最后一位呢？last指针前移，这不是真正意义上的删除元素，

否则的话，就是我们不需要删除的，那就指针后移，i+1，最后返回last+1

```python
def removeElement(nums, val):
    i, last = 0, len(nums)-1
    while i < last:
        if nums[i] == val:
            nums[i], nums[last] = nums[last], nums[i]
            last -= 1
        else:
            i += 1
    return last + 1

print(removeElement([0,1,2,2,3,0,4,2], 2))
```

**收获：**

让原地修改，那就不能初始化空list=[]，所以想双指针方法，换。

那就遍历列表吧，如果当前数==val那个数nums[i]==val，就和最后一个数换一下，然后删除(前移)一位，last-=1，

如果不相等，那指针后移，i+1

最后返回last+1，也就是删除元素之后的列表的长度。

## LeetCode-35-搜索插入位置

**题目：**

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
 输入: [1,3,5,6], 5  输出: 2  输入: [1,3,5,6], 2  输出: 1  输入: [1,3,5,6], 7  输出: 4  输入: [1,3,5,6], 0  输出: 0

**思路：**

如果target在array中，就返回index，不在的话返回nums[i]>target所在的index。

为了节省空间，先判断target是不是大于array最大的数，要是的话，直接返回array的长度len(array)

```python
def searchInsert(nums, val):
    if val > nums[len(nums)-1]:
        return len(nums)
    for i in range(len(nums)):
        if nums[i] >= val:
            return i

print(searchInsert([1, 3, 5, 6], 7))
```

**收获：**

因为是排序数组，所以为了节省空间，先看看最后一个数和目标值的大小比较，如果小的话，直接返回数组长度就可以了。

接下来就可以通过下标遍历nums，找到nums[i] >= val目标值的i，返回就行了。



## LeetCode-118杨辉三角-dp

**题目：**

给定一个非负整数 numRows，生成杨辉三角的前 *numRows* 行。

在杨辉三角中，每个数是它左上方和右上方的数的和。

**示例：**

```
输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```

**方法一：**

最简单的方法就是把每一行的数都generate出来，进行重复的套路，如果是首尾，那就是1，如果不是，那第i个数就是i-1行j-1的值和j的值相加之和

首先是一个空的result，然后每一行把生成的结果append到result里面去，要有多少行呢？就是numRows

首先先添加一个空的[]，然后再来一个循环，for j in range(i+1)，第一行有一个元素，第二行有两个元素，第三行有三个元素，…

```python
def triangle(x):
    result = [] 
    for i in range(x):
        result.append([])
        for j in range(i+1):
            if j in (0, i):  # 如果是这一行的首或者尾，也就是0和i
                result[i].append(1)
            else:
                result[i].append(result[i-1][j-1]+result[i-1][j])
    return result
print(triangle(5))

def triangle(x):
    dp = [[1]*(i+1) for i in range(x)]
    for i in range(2, x):
        for j in range(1, i):
            dp[i][j] = dp[i-1][j-1]+dp[i-1][j]
    return dp 
print(triangle(5))
```

**收获：**

>   方法一学到了判断一个数是list的首尾是if j in (0,i)
>
>   以及二维数组里对数据更改是result[i][]再取值。
>
>   方法二学会了先设置一个dp矩阵，然后再动态的更改里面的数据，注意索引的范围

## [Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)-dp

**题目：**

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

输入：n = 2   输出：2

输入：n = 7   输出：21

输入：n = 0   输出：1

**思路：动态规划**

此类求 多少种可能性 的题目一般都有 递推性质 ，即 f(n) 和 f(n-1)…f(1)之间是有联系的。

-   设跳上 n 级台阶有     f(n)种跳法。在所有跳法中，青蛙的最后一步只有两种情况： 跳上     1 级或 2 级台阶。

-   -   当为 1 级台阶： 剩 n-1 个台阶，此情况共有 f(n-1)种跳法；
    -   当为 2 级台阶： 剩 n-2 个台阶，此情况共有 f(n-2)种跳法。

-   f(n)为以上两种情况之和，即 f(n)=f(n-1)+f(n-2) ，以上递推性质为斐波那契数列。本题可转化为 求斐波那契数列第 n 项的值 ，与 面试题10-     I. 斐波那契数列 等价，唯一的不同在于起始数字不同。

-   -   青蛙跳台阶问题： f(0)=1 , f(1)=1 , f(2)=2；
    -   斐波那契数列问题： f(0)=0 , f(1)=1 , f(2)=1 。

```python
def stairs(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return stairs(n-1) + stairs(n-2)
print(stairs(0))

# 这样递归可以，但是复杂度太高，可以原地修改

def staris(n):
    f1 = 1
    f2 = 2
    for _ in range(n):
        f1, f2 = f2, f1 + f2
    return f1
print(stairs(7))
```

**收获：**

>动态规划法就是比较节省空间

## [Offer-11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

**题目：**

输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。

输入：[3,4,5,1,2]   输出：1   输入：[2,2,2,0,1]   输出：0

**思路：二分法**

二分法 解决，其可将 遍历法 的 线性级别 时间复杂度降低至 对数级别。

-   初始化： 声明     i, j 双指针分别指向 nums数组左右两端；

-   循环二分： 设     m = (i + j) / 2为每次二分的中点（ “/” 代表向下取整除法，因此恒有 i≤m<j ），可分为以下三种情况：

-   -   当 nums[m] > nums[j] 时： m 一定在 左排序数组 中，即旋转点 x 一定在      [m+1,j] 闭区间内，因此执行i=m+1；
    -   当 nums[m]<nums[j] 时： m 一定在 右排序数组 中，即旋转点 x 一定在[i,m]      闭区间内，因此执行j=m；
    -   当 nums[m]=nums[j] 时： 无法判断      m 在哪个排序数组中，即无法判断旋转点 x 在      [i,m] 还是[m+1,j] 区间中。解决方案： 执行j=j−1 缩小判断范围，分析见下文。

-   返回值： 当i=j时跳出二分循环，并返回旋转点的值nums[i] 即可。

```python
def minArray(numbers):
    i, j = 0, len(numbers) - 1
    while i < j:
        m = (i + j) // 2
        if numbers[m] > numbers[j]:
            i = m + 1
        elif numbers[m] < numbers[j]:
            j = m
        else:
            j -= 1
    return numbers[i]

print(minArray([2, 2, 2, 0, 1]))
```

**收获：**

>   取中间的那个数 (i+j) // 2，而且是向下取整
>   通过比较中间的数值大小和最右边数的大小判断最小值在哪个区间
>   最后通过减小j-=1来缩小范围

## [Offer-17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

**题目：**

比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

输入: n = 1   输出: [1,2,3,4,5,6,7,8,9]

**思路：**

最大的 1 位数是 9 ，最大的 2 位数是 99 ，最大的 3 位数是 999 。则可推出公式：end = 10^n - 1

那么遍历每一个数存到res里就可以了！

```python
def printnums(n):
    return list(range(1, 10 ** n))
print(printnums(2))
```

**收获：**

>学到了一个数的多少次方是，10 ** n

##     [Offer-21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

**题目：**

输入：nums = [1,2,3,4]   输出：[1,3,2,4] or [3,1,2,4] √

**思路：**

双指针，换，判断一个数是否是奇数可以num&1==1来判断。

1.   指针 i 从左向右寻找偶数；

2.   指针 j 从右向左寻找奇数；

3.   将 偶数 nums[i] 和 奇数 nums[j] 交换。

-   复杂度分析：

-   -   时间复杂度 O(N) ：      N 为数组 nums 长度，双指针 i, j 共同遍历整个数组。
    -   空间复杂度 O(1) ： 双指针 i, j 使用常数大小的额外空间。

```python
def exchange(nums):
    i, j = 0, len(nums) - 1
    while i < j:
        while i < j and nums[i] & 1 == 1:
            i += 1
        while i < j and nums[j] & 1 == 0:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    return nums
print([1,2,3,4])
```

**收获：**

>   学会了判断奇偶数用 & 1
>   奇数 & 1 = 1
>   偶数 & 1 = 0

## [Offer-40.最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

**题目：**

输入：arr = [3,2,1], k = 2  输出：[1,2] 或者 [2,1]

输入：arr = [0,1,2,1], k = 1   输出：[0]

-   基础排序算法总结

-   -   交换类：冒泡排序、快速排序
    -   选择类：简单选择排序、快速排序
    -   插入类：直接插入排序、shell 排序
    -   归并类：归并排序

### 1. 交换类排序 – 冒泡排序

-   思想：从无序序列头部开始，进行两两比较，根据大小交换位置，直到最后将最大（小）的数据元素交换到了无序队列的队尾，从而成为有序序列的一部分；下一次继续这个过程，直到所有数据元素都排好序。
-   时间复杂度：O(n2),空间复杂度 O(1)
-   稳定性： 稳定

```python
class Solution:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]
    def bubble_sort(self, nums):
        for i in range(1, len(nums)):
            flag = False
            for j in range(0, len(nums)-i):
                if nums[j] > nums[j+1]:
                    flag = True
                    self.swap(nums, j, j+1)
            if not flag:
                return nums
        return nums
    def getLeastNumbers(self, arr, k):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr
        return self.bubble_sort(arr)[:k]

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```



### 2. 交换类排序 – 快速排序

-   思想：分别从初始序列“6 1 2 7 9 3 4 5 10     8”两端开始“探测”。先从右往左找一个小于 6 的数，再从左往右找一个大于 6 的数，然后交换他们。这里可以用两个变量 i 和 j，分别指向序列最左边和最右边。我们为这两个变量起个好听的名字“哨兵 i”和“哨兵 j”。刚开始的时候让哨兵 i 指向序列的最左边（即 i=1），指向数字 6。让哨兵 j 指向序列的最右边（即 j=10），指向数字 8。

首先哨兵 j 开始出动。因为此处设置的基准数是最左边的数，所以需要让哨兵 j 先出动，这一点非常重要（请自己想一想为什么）。哨兵 j 一步一步地向左挪动（即 j–），直到找到一个小于 6 的数停下来。接下来哨兵 i 再一步一步向右挪动（即 i++），直到找到一个数大于 6 的数停下来。最后哨兵 j 停在了数字 5 面前，哨兵 i 停在了数字 7 面前。

现在交换哨兵 i 和哨兵 j 所指向的元素的值。交换之后的序列如下。6 1 2 5 9 3 4 7 10 8

到此，第一次交换结束。接下来开始哨兵 j 继续向左挪动（再友情提醒，每次必须是哨兵 j 先出发）。他发现了 4（比基准数 6 要小，满足要求）之后停了下来。哨兵 i 也继续向右挪动的，他发现了 9（比基准数 6 要大，满足要求）之后停了下来。此时再次进行交换，交换之后的序列如下。 6 1 2 5 4 3 9 7 10 8

第二次交换结束，“探测”继续。哨兵 j 继续向左挪动，他发现了 3（比基准数 6 要小，满足要求）之后又停了下来。哨兵 i 继续向右移动，糟啦！此时哨兵 i 和哨兵 j 相遇了，哨兵 i 和哨兵 j 都走到 3 面前。说明此时“探测”结束。我们将基准数 6 和 3 进行交换。交换之后的序列如下。 3 1 2 5 4 6 9 7 10 8

每次排序的时候设置一个基准点，将小于等于基准点的数全部放到基准点的左边，将大于等于基准点的数全部放到基准点的右边。

-   时间复杂度：O(nlogn),空间复杂度 O(nlogn)
-   稳定性： 不稳定

```python
class Solution:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def partition(self, nums, left, right):
        pivot = left
        i = j = pivot + 1

        while j <= right:
            if nums[j] <= nums[pivot]:
                self.swap(nums, i, j)
                i += 1
            j += 1
        self.swap(nums, pivot, i - 1)
        return i - 1

    def quick_sort(self, nums, left, right):
        if left < right:
            pivot = self.partition(nums, left, right)
            self.quick_sort(nums, left, pivot - 1)
            self.quick_sort(nums, pivot + 1, right)

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr
        self.quick_sort(arr, 0, len(arr) - 1)
        return arr[:k]

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```





### 3. 选择类排序 – 简单选择排序

-   思想：每一趟从待排序的数据元素中选择最小（或最大）的一个元素作为首元素，直到所有元素排完为止
-   时间复杂度：O(n2),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    # 循环迭代地将后面未排序序列最小值交换到前面有序序列末尾
    def select_sort(self, nums):
        for i in range(len(nums) - 1):
            min_index = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[min_index]:
                    min_index = j
            if min_index != i:
                self.swap(nums, i, min_index)
        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr

        return self.select_sort(arr)[:k]

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```





### 4. 选择类排序 – 堆排序

-   [思想](https://www.cnblogs.com/chengxiao/p/6129630.html)：将待排序序列构造成一个大顶堆，此时，整个序列的最大值就是堆顶的根节点。将其与末尾元素进行交换，此时末尾就为最大值。然后将剩余n-1个元素重新构造成一个堆，这样会得到n个元素的次小值。如此反复执行，便能得到一个有序序列了
-   时间复杂度：O(nlogn),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def heapify(self, nums, i, size):
        left, right = 2 * i + 1, 2 * i + 2
        largest = i
        if left < size and nums[left] > nums[largest]:
            largest = left
        if right < size and nums[right] > nums[largest]:
            largest = right
        if largest != i:
            self.swap(nums, i, largest)
            self.heapify(nums, largest, size)

    def build_heap(self, nums, k):
        # 从最后一个非叶子节点开始堆化
        for i in range(k // 2 - 1, -1, -1):
            self.heapify(nums, i, k)

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr
        heap = arr[:k]
        self.build_heap(heap, k)
        for i in range(k, len(arr)):
            if arr[i] < heap[0]:
                heap[0] = arr[i]
                self.heapify(heap, 0, k)
        return heap

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```

### 5. 插入类排序 – 直接插入排序

-   思想：把n个待排序的元素看成为一个有序表和一个无序表。开始时有序表中只包含1个元素，无序表中包含有n-1个元素，排序过程中每次从无序表中取出第一个元素，将它插入到有序表中的适当位置，使之成为新的有序表，重复n-1次可完成排序过程。
-   时间复杂度：O(n2),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution:
    def insert_sort(self, nums):
        for i in range(1, len(nums)):
            cur_value = nums[i]
            pos = i
            while pos >= 1 and nums[pos - 1] > cur_value:
                nums[pos] = nums[pos - 1]
                pos -= 1
            nums[pos] = cur_value
        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr
        return self.insert_sort(arr)[:k]

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```





### 6. 插入类排序 – shell 排序

-   [思想](https://www.cnblogs.com/chengxiao/p/6104371.html)：希尔排序是把记录按下标的一定增量分组，对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的关键词越来越多，当增量减至1时，整个文件恰被分成一组，算法便终止。
-   时间复杂度：O(n1.3),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution:
    # 分组插入排序
    def insert_sort_gap(self, nums, start, gap):
        for i in range(start + gap, len(nums), gap):
            cur_value = nums[i]
            pos = i
            while pos >= gap and nums[pos - gap] > cur_value:
                nums[pos] = nums[pos - gap]
                pos -= gap
            nums[pos] = cur_value

    def shell_sort(self, nums):
        gap = len(nums) // 2
        while gap > 0:
            for i in range(gap):
                self.insert_sort_gap(nums, i, gap)
            gap //= 2
        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr
        return self.shell_sort(arr)[:k]

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```





### 7. 归并类排序 – 归并排序

-   [思想](https://www.cnblogs.com/chengxiao/p/6194356.html)：归并排序（MERGE-SORT）是利用归并的思想实现的排序方法，该算法采用经典的分治（divide-and-conquer）策略（分治法将问题分(divide)成一些小的问题然后递归求解，而治(conquer)的阶段则将分的阶段得到的各答案”修补”在一起，即分而治之)。
-   时间复杂度：O(nlogn),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution:
    def merge(self, l1, l2):
        res = []
        i, j = 0, 0
        while i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                res.append(l1[i])
                i += 1
            else:
                res.append(l2[j])
                j += 1
        if i == len(l1):
            res.extend(l2[j:])
        else:
            res.extend(l1[i:])
        return res

    def merge_sort(self, nums):
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left = self.merge_sort(nums[:mid])
        right = self.merge_sort(nums[mid:])
        return self.merge(left, right)

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0:
            return []
        if len(arr) <= k:
            return arr
        return self.merge_sort(arr)[:k]

s = Solution()
print(s.getLeastNumbers([3, 2, 1, 4, 5, 6], 4))
```

