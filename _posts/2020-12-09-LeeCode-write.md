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

JavaScript：

```js
var reverse = function(x) {
    let result = 0;
    while(x !== 0) {
        result = result * 10 + x % 10;
        x = (x / 10) | 0;
    }
    return (result | 0) === result ? result : 0;
};
```

收获：

-   `result * 10 + x % 10` 取出末位 `x % 10`（负数结果还是负数，无需关心正负），拼接到 `result` 中。
-   `x / 10` 去除末位，`| 0` 强制转换为32位有符号整数。
-   通过 `| 0` 取整，无论正负，只移除小数点部分（正数向下取整，负数向上取整）。
-   `result | 0` 超过32位的整数转换结果不等于自身，可用作溢出判断。

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

JavaScript：

```js
var longestCommonPrefix = function(strs) {
    if(strs.length == 0) 
        return "";
    let ans = strs[0];
    for(let i =1;i<strs.length;i++) {
        let j=0;
        for(;j<ans.length && j < strs[i].length;j++) {
            if(ans[j] != strs[i][j])
                break;
        }
        ans = ans.substr(0, j);
        if(ans === "")
            return ans;
    }
    return ans;
};
```

标签：链表
当字符串数组长度为 0 时则公共前缀为空，直接返回
令最长公共前缀 ans 的值为第一个字符串，进行初始化
遍历后面的字符串，依次将其与 ans 进行比较，两两找出公共前缀，最终结果即为最长公共前缀
如果查找过程中出现了 ans 为空的情况，则公共前缀不存在直接返回
时间复杂度：O(s)，s 为所有字符串的长度之和

## LeetCode-[125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

**题目：**

```
输入: "A man, a plan, a canal: Panama"   输出: true
输入: "race a car"    输出: false
```

**方法一**：

最好的方法是有两个指针，一个指向头一个指向尾，每读一个往中间进一步，看是不是一样的，当走到中间或者右边的指针走到左边的时候，对比就完成了。

需要有while i < j and not s[i].isalnum()这一行判断当前的s[i]是不是字母，因为会有空格和标点的存在！

```python
def isPalindrome(s):
    i, j = 0, len(s) -1
    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        if s[i].lower() != s[j].lower():
            return False
        i += 1
        j -= 1
    return True

print(isPalindrome("A man, a plan, a canal: Panama"))
```

**收获：**

两个指针首尾比较嘛。

直到找到第一个是字母的写法while i>j and s[i].isalnum(): i += 1

直到找到最后一个是字母的写法while i<j and s[j].isalnum():j-=1

然后进行比较，记得加.lower() 

如果不相等直接返回false，否则的话i，j分别往里走一步然后继续比较。

JavaScript:

思路：

-   `\W`匹配，非数字、非字母和非下划线，`[\W|_]`，加上下划线
-   如果`匹配到`，左指针左移，右指针右移。只要左右指针指到不等的数字或字母，返回`false`

双指针

```js
var isPalindrome = function(s, l = 0, r = s.length - 1) {  
    while(l < r) {
        if (/[\W|_]/.test(s[l]) && ++l) continue
        if (/[\W|_]/.test(s[r]) && r--) continue
        if (s[l].toLowerCase() !== s[r].toLowerCase()) return false
        l++, r--
    }
    return true
};
```

先正则替换，再双指针

```js
var isPalindrome = function(s, l = -1, r) {  
    r = (s = s.replace(/[\W|_]/g, '').toLowerCase()).length
    while(++l < --r) if (s[l] !== s[r]) return false
    return true
};
```

## [Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

**题目：**

输入：s = "We are happy."  输出："We%20are%20happy."

**思路一：**直接遍历

新建字符串，遍历逐个相加，如果是空格就替换成%20，不是的话加到到新建的字符串后面

```python
def replaceSpace(s):
    res = ''
    for c in s:
        if c == ' ':
            res += '%20'
        else:
            res += c
    return res

print(replaceSpace("We are happy."))
```

-   时间复杂度 O(N) ：遍历字符串。
-   空间复杂度 O(N) ：字符串的长度。

**思路二：**join函数

```python
def replaceSpace(s):
    li = s.split(' ')
    return '%20'.join(li)

print(replaceSpace("We are happy."))
```

**思路三**：replace函数

```python
def replaceSpace(s):
    return s.replace(' ', '%20')

print(replaceSpace("We are happy."))
```

**JavaScript:**

**解题思路**
 首先判断输入是否合法，参数是字符串类型，字符串长度不能太长。
 再通过split(' ')将空格隔开的单词变为字符串数组中的数组项
 最后通过join('%20')将各个数组项，也就是单词，连接起来完成空格的替换。

**代码**

```js
/**
 * @param {string} s
 * @return {string}
 */
var replaceSpace = function(s) {
      if (typeof s == "string" && s.length >= 0 && s.length <= 10000) {
        return s.split(' ').join('%20');
      }
      return '';
};

正则
var replaceSpace = function(s) {
    return s.replace(/ /g, "%20");
};
```



## [Offer-58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/) [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

**题目：**

输入: "the sky is blue"   输出: "blue is sky the"    输入: "  hello world!  "   输出: "world! hello"   输入: "a good  example"   输出: "example good a"

**方法一：双指针**

算法解析：

-   倒序遍历字符串 s ，记录单词左右索引边界 i , j ；
-   每确定一个单词的边界，则将其添加至单词列表 res ；
-   最终，将单词列表拼接为字符串，并返回即可。

复杂度分析：

-   时间复杂度 O(N) ： 其中 N 为字符串 s 的长度，线性遍历字符串。
-   空间复杂度 O(N) ： 新建的 list(Python) 或     StringBuilder(Java) 中的字符串总长度 ≤N ，占用 O(N) 大小的额外空间。

**思路一：双指针**

```python
def reverseWords(s):
    s = s.strip()
    i = j = len(s) - 1
    res = []
    while i >= 0:
        while i>=0 and s[i]!= ' ':
            i -= 1
        res.append(s[i+1:j+1])
        while s[i]== ' ':
            i -= 1
        j = i
    return ' '.join(res)

print(reverseWords('the sky is blue'))
```

**收获：**

逆序翻转字符串里的单词，那就倒着数每一个字符，只要不是空格，那就继续往前走，直到遇到空格位置。

用双指针，全指向字符串的末尾，用i，j来卡一个完整的单词。

找到空格以后就可以把i指针再往前走一个，然后给了j，作为新的字符串的末尾。

**思路二：用函数**

```python
def reverseWords(s):
    return ' '.join(s.strip().split(' ')[::-1])

print(reverseWords('the sky is blue'))
```

JavaScript：

解法：先用trim()把字符串两端空格去掉，split(' ')把字符串切割成以空格为界限的单词块，filter()过滤掉数组中的纯空格，reverse()进行数组反转，join(' ')把数组变成中间只带一个空格的字符串


```js
var reverseWords = function (s) {
    var str = s.trim().split(' ').filter(item => item!='').reverse().join(' ')
    console.log(str)
};
```

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

JavaScript:

思路
首先将所有的组合可能性列出并添加到哈希表中
然后对字符串进行遍历，由于组合只有两种，一种是 1 个字符，一种是 2 个字符，其中 2 个字符优先于 1 个字符
先判断两个字符的组合在哈希表中是否存在，存在则将值取出加到结果 ans 中，并向后移2个字符。不存在则将判断当前 1 个字符是否存在，存在则将值取出加到结果 ans 中，并向后移 1 个字符
遍历结束返回结果 ans

```js
/**
 * @param {string} s
 * @return {number}
 */
var romanToInt = function(s) {
    const map = {
        I : 1,
        IV: 4,
        V: 5,
        IX: 9,
        X: 10,
        XL: 40,
        L: 50,
        XC: 90,
        C: 100,
        CD: 400,
        D: 500,
        CM: 900,
        M: 1000
    };
    let ans = 0;
    for(let i = 0;i < s.length;) {
        if(i + 1 < s.length && map[s.substring(i, i+2)]) {
            ans += map[s.substring(i, i+2)];
            i += 2;
        } else {
            ans += map[s.substring(i, i+1)];
            i ++;
        }
    }
    return ans;
};
```

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

JavaScript

判断括号的有效性可以使用「栈」这一数据结构来解决。

我们对给定的字符串 ss 进行遍历，当我们遇到一个左括号时，我们会期望在后续的遍历中，有一个相同类型的右括号将其闭合。由于后遇到的左括号要先闭合，因此我们可以将这个左括号放入栈顶。

当我们遇到一个右括号时，我们需要将一个相同类型的左括号闭合。此时，我们可以取出栈顶的左括号并判断它们是否是相同类型的括号。如果不是相同的类型，或者栈中并没有左括号，那么字符串 ss 无效，返回 \text{False}False。为了快速判断括号的类型，我们可以使用哈希映射（HashMap）存储每一种括号。哈希映射的键为右括号，值为相同类型的左括号。

在遍历结束后，如果栈中没有左括号，说明我们将字符串 ss 中的所有左括号闭合，返回 \text{True}True，否则返回 \text{False}False。

注意到有效字符串的长度一定为偶数，因此如果字符串的长度为奇数，我们可以直接返回 \text{False}False，省去后续的遍历判断过程。

```js
var isValid = function(s) {
    const n = s.length;
    if (n % 2 === 1) {
        return false;
    }
    const pairs = new Map([
        [')', '('],
        [']', '['],
        ['}', '{']
    ]);
    const stk = [];
    s.split('').forEach(ch => {
        if (pairs.has(ch)) {
            if (!stk.length || stk[stk.length - 1] !== pairs.get(ch)) {
                return false;
            }
            stk.pop();
        } 
        else {
            stk.push(ch);
        }
    });
    return !stk.length;
};
```



## [Offer-50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

**题目：**

s = "abaccdeff"   返回 "b"

s = ""   返回 " "

**思路1：**

1：哈希表

\1.   先遍历字符串 s ，统计各字符数量是否 > 1，dic[c] = not c in dic，dic = {'a': False, 'b': True, 'c': False, 'd': True, 'e': True, 'f': False}

\2.   再遍历字符串 s ，在哈希表中找到首个 “数量为 1 的字符”，并返回，if dic[c]，c

复杂度分析：

-   时间复杂度 O(N) ： N 为字符串 s 的长度；需遍历 s 两轮，使用 O(N) ；HashMap 查找操作的复杂度为 O(1) ；
-   空间复杂度 O(1) ： 由于题目指出 s 只包含小写字母，因此最多有 26 个不同字符，HashMap 存储需占用 O(26) = O(1) 的额外空间。

```python
def firstUniqChar(s):
    dic = {}
    for c in s:
        dic[c] = not c in dic
    for c in s:
        if dic[c]:
            return c
    return ''

print(firstUniqChar('anaccdeff'))
```

**收获：**

键值对是字符：在不在字典中出现可以是dic[c] = not c in dic

如果出现，可以是if dic[c]:

**思路2：**

相比于方法一，方法二减少了第二轮遍历的循环次数。当字符串很长（重复字符很多）时，方法二则效率更高。

时间和空间复杂度均与 “方法一” 相同，而具体分析：方法一 需遍历 s 两轮；方法二 遍历 s 一轮，遍历 dic 一轮（ dic 的长度不大于 26 ）。

```python
from collections import OrderedDict
def firstUniqChar(s):
    dic = OrderedDict()
    for c in s:
        dic[c] = not c in dic
    for k,v in dic.items():
        if v:
            return k
    return ''
print(firstUniqChar('anaccdeff'))
```

**收获：**

orderdic的使用

javascript

Set + 正则

```js
var firstUniqChar = function(s) {
  for (let char of new Set(s)) {
    if (s.match(new RegExp(char, 'g')).length === 1) {
      return char;
    }
  }
  return ' ';
};
```



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

Javascript:

思路

-   用一个读指针，一个写指针遍历数组。
-   遇到重复的元素 `读指针` 就继续前移。
-   遇到不同的元素 `写指针` 就前移一步，写入那个元素。

```js
/**
 * @param {number[]} nums
 * @return {number}
 */
var removeDuplicates = function (nums) {
    let p1 = 0,
        p2 = 0;

    while (p2 < nums.length) {
        if (nums[p1] != nums[p2]) {
            p1++;
            nums[p1] = nums[p2];
        }
        p2++;
    }
    return p1 + 1;
};
```



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

javascript

第一种思路
标签：拷贝覆盖
主要思路是遍历数组 nums，每次取出的数字变量为 num，同时设置一个下标 ans
在遍历过程中如果出现数字与需要移除的值不相同时，则进行拷贝覆盖 nums[ans] = num，ans 自增 1
如果相同的时候，则跳过该数字不进行拷贝覆盖，最后 ans 即为新的数组长度
这种思路在移除元素较多时更适合使用，最极端的情况是全部元素都需要移除，遍历一遍结束即可
时间复杂度：O(n)，空间复杂度：O(1)

```js
/**
 * @param {number[]} nums
 * @param {number} val
 * @return {number}
 */
var removeElement = function(nums, val) {
    let ans = 0;
    for(const num of nums) {
        if(num != val) {
            nums[ans] = num;
            ans++;
        }
    }
    return ans;
};
```

第二种思路
标签：交换移除
主要思路是遍历数组 nums，遍历指针为 i，总长度为 ans
在遍历过程中如果出现数字与需要移除的值不相同时，则i自增1，继续下一次遍历
如果相同的时候，则将 nums[i]与nums[ans-1] 交换，即当前数字和数组最后一个数字进行交换，交换后就少了一个元素，故而 ans 自减 1
这种思路在移除元素较少时更适合使用，最极端的情况是没有元素需要移除，遍历一遍结束即可
时间复杂度：O(n)O(n)，空间复杂度：O(1)O(1)

```js
/**
 * @param {number[]} nums
 * @param {number} val
 * @return {number}
 */
var removeElement = function(nums, val) {
    let ans = nums.length;
    for (let i = 0; i < ans;) {
        if (nums[i] == val) {
            nums[i] = nums[ans - 1];
            ans--;
        } else {
            i++;
        }
    }
    return ans;
};
```



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

javascript

```js
var searchInsert = function(nums, target) {
    const n = nums.length;
    let left = 0, right = n - 1, ans = n;
    while (left <= right) {
        let mid = ((right - left) >> 1) + left;
        if (target <= nums[mid]) {
            ans = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return ans;
};
```



## LeetCode-66-加一

**题目：**

输入: [1,2,3]  输出: [1,2,4]
 输入: [4,3,2,1]  输出: [4,3,2,2]

**思路：**

直接从最后一位数起，如果是9，那就置为0，如果不是9，就是+1返回，

如果遍历完了全都是9，那就手动把第一个变为1，然后结尾再添加一个0，注意添加的位置是在for循环结束以后。

```python
def plusOne(digits):
    for i in reversed(range(len(digits))):
        if digits[i] == 9:
            digits[i] = 0
        else: 
            digits[i] += 1
            return digits
    digits[0] = 1
    digits.append(0)
    return digits
        
print(plusOne([1,1,9,9]))
```

**收获：**

倒着遍历list，reversed(range(len(digits)))

还有，分两种情况，要有2个return。

JavaScript

思路
标签：数组遍历
这道题需要整理出来有哪几种情况，在进行处理会更舒服
末位无进位，则末位加一即可，因为末位无进位，前面也不可能产生进位，比如 45 => 46
末位有进位，在中间位置进位停止，则需要找到进位的典型标志，即为当前位 %10后为 0，则前一位加 1，直到不为 0 为止，比如 499 => 500
末位有进位，并且一直进位到最前方导致结果多出一位，对于这种情况，需要在第 2 种情况遍历结束的基础上，进行单独处理，比如 999 => 1000
在下方的 Java 和 JavaScript 代码中，对于第三种情况，对其他位进行了赋值 0 处理，Java 比较 tricky 直接 new 数组即可，JavaScript 则使用了 ES6 语法进行赋值
时间复杂度：O(n)

```js
/**
 * @param {number[]} digits
 * @return {number[]}
 */
var plusOne = function(digits) {
    const len = digits.length;
    for(let i = len - 1; i >= 0; i--) {
        digits[i]++;
        digits[i] %= 10;
        if(digits[i]!=0)
            return digits;
    }
    digits = [...Array(len + 1)].map(_=>0);;
    digits[0] = 1;
    return digits;
};
```



## LeetCode-[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

**题目：**

nums1 = [1,2,3,0,0,0], m = 3

nums2 = [2,5,6],    n = 3

输出: [1,2,2,3,5,6]

思路：

从后往前合并。

nums1[m-1]是第一个list除0外最后一个数，

nums2[n-1]是第二个list最后一个数，

nums1[m+n-1]是第一个list最后一个数

比较大小，如果是第二个list的数比较大，那就把它放到第一个list最后的位置，然后第二个list的指针前移，

否则的话第一个list最后一个数与最后一位的数换一下，然后第一个list的指针前移，

一直循环，直到第一个list遍历完，m=0，然后剩下的前n个数复制到第一个list里。

```python
def merge(nums1, m, nums2, n):
    while m>0 and n>0:
        if nums1[m-1] < nums2[n-1]:
            nums1[m+n-1] = nums2[n-1]
            n -= 1
        else:
            nums1[m+n-1], nums1[m-1] = nums1[m-1],nums1[m+n-1]
            m -= 1
        if m == 0 and n > 0:
            nums1[:n] = nums2[:n]
    return nums1

print(merge([2,3,0,0,0],2,[2,5,6],3))
```

**收获：**

思路还是挺清晰的，明确三个变量是什么

nums1[m-1]是第一个list除0外最后一个数，

nums2[n-1]是第二个list最后一个数，

nums1[m+n-1]是第一个list最后一个数

然后比较大小，第一个list是换，第二个list是放置，

最后别忘了剩下的数要复制到前边去。

javascript

思路
标签：从后向前数组遍历
因为 nums1 的空间都集中在后面，所以从后向前处理排序的数据会更好，节省空间，一边遍历一边将值填充进去
设置指针 len1 和 len2 分别指向 nums1 和 nums2 的有数字尾部，从尾部值开始比较遍历，同时设置指针 len 指向 nums1 的最末尾，每次遍历比较值大小之后，则进行填充
当 len1<0 时遍历结束，此时 nums2 中海油数据未拷贝完全，将其直接拷贝到 nums1 的前面，最后得到结果数组
时间复杂度：O(m+n)O(m+n)

```js
/**
 * @param {number[]} nums1
 * @param {number} m
 * @param {number[]} nums2
 * @param {number} n
 * @return {void} Do not return anything, modify nums1 in-place instead.
 */
var merge = function(nums1, m, nums2, n) {
    let len1 = m - 1;
    let len2 = n - 1;
    let len = m + n - 1;
    while(len1 >= 0 && len2 >= 0) {
        // 注意--符号在后面，表示先进行计算再减1，这种缩写缩短了代码
        nums1[len--] = nums1[len1] > nums2[len2] ? nums1[len1--] : nums2[len2--];
    }
    function arrayCopy(src, srcIndex, dest, destIndex, length) {
        dest.splice(destIndex, length, ...src.slice(srcIndex, srcIndex + length));
    }
    // 表示将nums2数组从下标0位置开始，拷贝到nums1数组中，从下标0位置开始，长度为len2+1
    arrayCopy(nums2, 0, nums1, 0, len2 + 1);
};
```



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

javascript

```js
var generate = function(numRows) {
    const ret = [];
    for (let i = 0; i < numRows; i++) {
        const row = new Array(i + 1).fill(1);
        for (let j = 1; j < row.length - 1; j++) {
            row[j] = ret[i - 1][j - 1] + ret[i - 1][j];
        }
        ret.push(row);
    }
    return ret;
};
```



## LeetCode-[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

**题目：**

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格，设计一个算法来计算你所能获取的最大利润。

输入: [7,1,5,3,6,4]  输出: 5  解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。

输入: [7,6,4,3,1]   输出: 0  解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

**思路：**

需要找到一个最好的卖出点：

即当前价格和最小价格差值是最大的时候，max(max_profit, price-min_price)，

用min(min_price, price)来记录每一轮的最小价格是多少，

用max(max_profit, price-min_price)，来记录最大差值是多少。

最后返回预先定义好的max_profit。

```python
def maxProfit(prices):
    max_profit, min_price = 0, float('inf')
    for price in prices:
        min_price = min(price, min_price)
        max_profit = max(max_profit, price-min_price)
    return max_profit

print(maxProfit([7,1,5,3,6,4]))
```

**收获：**

一个无穷大的数可以用float("inf")表示，

找到数组里最小的数可以用min(price, min_price)来比较，

找到差值最大的数可以用max(max_prifit, price-min_price)来找到

javascript

```js
/**
 * @param {number[]} prices
 * @return {number}
 */
var maxProfit = function(prices) {
    /*
        思路一 双层遍历 O(n^2) 
            a 外层遍历i 0~prices.length - 1
            b 内层遍历j i + 1~prices.length - 1
                c 找出大于当前项目 prices[i] 并 卖出 并 更新最大值
            d 输出结果
     */   
    if (!prices || !prices.length) return 0
    const len = prices.length
    let max = 0, cur = 0, next = 0
    for (let i = 0; i < len; i++) {
        cur = prices[i]
        for (let j = i + 1; j < len; j++) {
            next = prices[j]
            if (next > cur) {
                max = Math.max(max, next - cur)
            }
        }
    }
    return max

    /* 
        思路二 DP  Time: O(n) + Space: O(n)
        dp[i] 前i天卖出的最大利润
        min : prices 前i项中的最小值
        prices[i] - min: 当前位置卖出可得最大利润
        dp[i - 1] : 前i-1项目卖出可得的最大利润
        [7, 1, 5, 3, 6, 4] => dp[i] = Math.max( dp[i - 1], prices[i] - min )
        [7]                [0, 0, 0, 0, 0, 0]
        [7, 1]             [0, 0, 0, 0, 0, 0]
        [7, 1, 5]          [0, 0, 4, 0, 0, 0]
        [7, 1, 5, 3]       [0, 0, 4, 4, 0, 0]
        [7, 1, 5, 3, 6]    [0, 0, 4, 4, 5, 0]
        [7, 1, 5, 3, 6, 4] [0, 0, 4, 4, 5, 5]

        输出结果 dp[len - 1]
    */ 
    if (!prices || !prices.length) return 0
    const len = prices.length, dp = new Array(len).fill(0)
    let min = prices[0] // 前i项的最小值
    for (let i = 1, price; i < len; i++) {
        price = prices[i]
        min = Math.min(min, price)
        dp[i] = Math.max(dp[i - 1], price - min )
    }
    return dp[len - 1]
    
    /* 
        思路三 DP + 常量级变量 min max Time - O(n) + Space - O(1)
        精简 我们只关心 max 与 min 故不需要再构建dp 数组
    */
    if (!prices || !prices.length) return 0
    let min = Number.MAX_SAFE_INTEGER, max = 0
    for (let i = 0, price; i < prices.length; i++) {
        price = prices[i]
        min = Math.min(min, price)
        max = Math.max(max, price - min)
    }
    return max
};

var maxProfit = function(prices) {
    /* 
        思路四 极简版 一行代码 巧用reduce + [min, max] 本质上是思路三的一种简写方法 
        虽然 只有一行代码 但是 可读性 与 推展性 不高 生产环境的话还是推荐 思路三
        prices.reduce((p, v) => [
            Math.min(p[0], v), // 更新最小值 
            Math.max(p[1], v - p[0] ) // 更新最大值
        ], [Number.MAX_SAFE_INTEGER, 0])[1]
    */
    return prices.reduce((p, v) => [Math.min(p[0], v), Math.max(p[1], v - p[0]) ], [Number.MAX_SAFE_INTEGER, 0])[1]
}
```





## LeetCode-[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

**题目：**

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

输入: [7,1,5,3,6,4]  输出: 7  解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

输入: [1,2,3,4,5]   输出: 4  解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。

输入: [7,6,4,3,1]   输出: 0  解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

**思路：**

如果后面的数要是比前面的数大，就把差值加到total里。

可以用if prices[i] > prices[i-1]: total += prices[i]-prices[i-1]，得注意range是从range(1,len(prices)) 1开始的。

```python
def maxProfit(prices):
    if len(prices) <= 1:
        return 0
    total = 0
    for i in range(1, len(prices)):
        if prices[i]>prices[i-1]:
            total += prices[i]-prices[i-1]
    return total

print(maxProfit([7,1,5,3,6,4]))
```

**收获：**

和思路一样，如果后面的数要是比前面的数大，就把差值加到total里。可以用if prices[i] > prices[i-1]: total += prices[i]-prices[i-1]表示。

javascript

```js
var maxProfit = function(prices) {
    let ans = 0;
    let n = prices.length;
    for (let i = 1; i < n; ++i) {
        ans += Math.max(0, prices[i] - prices[i - 1]);
    }
    return ans;
};
```



## LeetCode-[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

**题目：**

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

输入: [2,2,1]  输出: 1  输入: [4,1,2,1,2]  输出: 4

**思路：**

异或。两个相同的会是0，其余与0做异或是其本身，这样异或nums里的所有的数字，依次类推，单个的那个会留下来。

```python
def singleNumber(nums):
    res = 0
    for num in nums:
        res ^= num
    return res

print(singleNumber([4,1,2,1,2]))
```

**收获：**

异或，两个相同的数异或会是0，遍历一遍数组，单个的那个数会留下来。

javascript

```js
/**
 * @param {number[]} nums
 * @return {number}
 */
var singleNumber = function(nums) {
    let ans = 0;
    for(const num of nums) {
        ans ^= num;
    }
    return ans;
};
```



## [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

**题目：**

输入一个二维数组和一个整数，判断数组中是否含有该整数。

```
[
 [1,  4, 7, 11, 15],
 [2,  5, 8, 12, 19],
 [3,  6, 9, 16, 22],
 [10, 13, 14, 17, 24],
 [18, 21, 23, 26, 30]
]
```

**思路：左下角定位法**

定位左下角，按照行列索引确定matrix中的数字，

起始行列i，j行列坐标分别为左下角的len(matirx)-1和0，

然后开始循环，行索引i要逐渐的减小，直到i>=0，列索引要逐渐的增大，直到j<len(matix[0])，

判断matrix中的数字[i] [j]与要找的数字大小的关系，进行行列索引的++–，找到了返回True，没有找到，出了循环返回False。

```python
def findNumberIn2DArray(matrix, target):
    i, j = len(matrix)-1, 0
    while i>=0 and j < len(matrix)-1:
        if matrix[i][j] > target:
            i -= 1
        elif matrix[i][j] < target:
            j += 1
        else:
            return True
    return False
    
print(findNumberIn2DArray([[1,  4, 7, 11, 15],[2,  5, 8, 12, 19],[3,  6, 9, 16, 22],[10, 13, 14, 17, 24],[18, 21, 23, 26, 30]],16))
```

**收获：**

左下角可以用len(matrix)-1, 0来表示，

循环的判断条件是*while* i>=0 and j < len(*matrix*)-1:

如果左下角数>target，那就行－1，

若果左下角数<target，那就列+1，

找到返回true，没找到返回false。



-   时间复杂度 O(M+N) ：其中，N 和 M 分别为矩阵行数和列数，此算法最多循环 M+N次。
-   空间复杂度 O(1) : i, j 指针使用常数大小额外空间。

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

## [Offer-29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/) [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

**题目：**

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]    输出：[1,2,3,6,9,8,7,4,5]

输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]   输出：[1,2,3,4,8,12,11,10,9,5,6,7]

解题思路：

根据题目示例 matrix = [[1,2,3],[4,5,6],[7,8,9]] 的对应输出 [1,2,3,6,9,8,7,4,5] 可以发现，顺时针打印矩阵的顺序是 “从左向右、从上向下、从右向左、从下向上” 循环。

复杂度分析：

-   时间复杂度 O(MN) ： M, N分别为矩阵行数和列数。

-   空间复杂度 O(1)： 四个边界 l , r , t , b 使用常数大小的 额外 空间（ res 为必须使用的空间）。

```python
def spiralOrder(matrix):
    if not matrix:
        return []
    l, r, t, b, res = 0, len(matrix[0])-1, 0, len(matrix)-1, []
    while True:
        for i in range(l, r + 1): 
            res.append(matrix[t][i]) # left to right                 
        t += 1
        if t > b: 
            break
        for i in range(t, b + 1): 
            res.append(matrix[i][r]) # top to bottom                 
        r -= 1
        if l > r: 
            break
        for i in range(r, l - 1, -1): 
            res.append(matrix[b][i]) # right to left                 
        b -= 1
        if t > b: 
            break
        for i in range(b, t - 1, -1): 
            res.append(matrix[i][l]) # bottom to top                 
        l += 1
        if l > r: 
            break
    return res
    
print(spiralOrder([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
```

**收获：**

好绕，没仔细看~，没用代码的语言梳理语言。

**思路2：**

```python

```

**收获：**

## [Offer-40.最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

**题目：**

输入：arr = [3,2,1], k = 2  输出：[1,2] 或者 [2,1]

输入：arr = [0,1,2,1], k = 1   输出：[0]

-   基础排序算法总结

-   -   交换类：冒泡排序、快速排序
    -   选择类：简单选择排序、快速排序
    -   插入类：直接插入排序、shell 排序
    -   归并类：归并排序

### 1. 交换类排序 – 冒泡排序√

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

**收获：**

写类方法，可以直接self.swap调用，而且不用写self

设置一个flag来设置是不是要交换

第一层for循环是range(1,len(nums))，因为是从第二个数字开始比较的。

第二层for循环是range(0,len(nums)-i)，是做了优化。

还有就是主函数getLeastNumbers那里，学习一下初试条件判断。

### 2. 交换类排序 – 快速排序√

-   思想：分别从初始序列“6 1 2 7 9 3 4 5 10     8”两端开始“探测”。先从右往左找一个小于 6 的数，再从左往右找一个大于 6 的数，然后交换他们。这里可以用两个变量 i 和 j，分别指向序列最左边和最右边。我们为这两个变量起个好听的名字“哨兵 i”和“哨兵 j”。刚开始的时候让哨兵 i 指向序列的最左边（即 i=1），指向数字 6。让哨兵 j 指向序列的最右边（即 j=10），指向数字 8。
-   每次排序的时候设置一个基准点，将小于等于基准点的数全部放到基准点的左边，将大于等于基准点的数全部放到基准点的右边。

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

### 7. 归并类排序 – 归并排序√

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

## [Offer-53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

**题目：**

统计一个数字在排序数组中出现的次数。

输入: nums = [5,7,7,8,8,10], target = 8   输出: 2   输入: nums = [5,7,7,8,8,10], target = 6   输出: 0

**思路：**

排序数组中的搜索问题，首先想到 二分法 解决。

所有数字target形成一个窗口，使用二分法分别找到左边界left和右边界right ，易得数字target的数量为right - left - 1。

**复杂度分析：**

时间复杂度 O(log N)： 二分法为对数级别复杂度。

空间复杂度 O(1)： 几个变量使用常数大小的额外空间。

```python
def search(nums, target):
     # 搜索右边界 right         
    i, j = 0, len(nums) - 1
    while i <= j:
        m = (i + j) // 2
        if nums[m] <= target:
            i = m + 1
        else:
            j = m - 1
    right = i
    # 若数组中无 target ，则提前返回         
    if j >= 0 and nums[j] != target: 
        return 0
    # 搜索左边界 left         
    i = 0
    while i <= j:
        m = (i + j) // 2
        if nums[m] < target:
            i = m + 1
        else:
            j = m - 1
    left = j
    return right - left - 1

print(search([5, 7, 7, 8, 8, 10], 8))
```

二分法还没想明白😓

```python
不用二分法，最直接的想法
def search(nums, target):
    res = 0
    for i in nums:
        if i == target:
            res += 1
    return res
print(search([5,7,7,8,8,10],8))
```

## [Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

**题目：**

输入: [0,1,3]  输出: 2   输入: [0,1,2,3,4,5,6,7,9]   输出: 8

**思路：**

二分法解决。

左子数组： nums[i] = i ；

右子数组： nums[i] ≠ i；

缺失的数字等于 “右子数组的首位元素” 对应的索引；

**复杂度分析：**

时间复杂度 O(log N)： 二分法为对数级别复杂度。

空间复杂度 O(1)： 几个变量使用常数大小的额外空间。

```python
def missingNumber(nums):
    i, j = 0, len(nums)-1
    while i <= j:
        m = int(i+j)//2
        if nums[m]==m:
            i = m+1
        else:
            j = m-1
    return i

print(missingNumber([0, 1, 2, 3, 4, 5, 6, 7, 9]))
```

**收获：**

if nums[m]==m，就是说下标上的数等于下标。

如果相等，就说明缺失的数还在右边，那i 就往右走一步，否则的话，j就往左走一步。

另外要注意，双指针，取中间值的时候，m=int(i+j)//2一定要写在while i<j 的循环里，要不然中间指针没有更新会陷入死循环。