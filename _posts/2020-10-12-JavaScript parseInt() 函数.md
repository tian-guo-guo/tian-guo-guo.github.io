---
layout:     post           # 使用的布局（不需要改）
title:      JavaScript parseInt() 函数           # 标题 
subtitle:   JavaScript parseInt() 函数           #副标题
date:       2020-10-12             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [JavaScript parseInt() 函数](https://www.runoob.com/jsref/jsref-parseint.html)

## 定义和用法

parseInt() 函数可解析一个字符串，并返回一个整数。

当参数 radix 的值为 0，或没有设置该参数时，parseInt() 会根据 string 来判断数字的基数。

当忽略参数 radix , JavaScript 默认数字的基数如下:

-   如果 string 以 "0x" 开头，parseInt() 会把 string 的其余部分解析为十六进制的整数。
-   如果 string 以 0 开头，那么 ECMAScript v3 允许 parseInt() 的一个实现把其后的字符解析为八进制或十六进制的数字。
-   如果 string 以 1 ~ 9 的数字开头，parseInt() 将把它解析为十进制的整数。

## 语法

parseInt(string, radix)



| 参数   | 描述                                                 |
| :----- | :--------------------------------------------------- |
| string | 必需。要被解析的字符串。                             |
| radix  | 可选。表示要解析的数字的基数。该值介于 2 ~ 36 之间。 |



------

## 浏览器支持

![Internet Explorer](https://www.runoob.com/images/compatible_ie.gif)![Firefox](https://www.runoob.com/images/compatible_firefox.gif)![Opera](https://www.runoob.com/images/compatible_opera.gif)![Google Chrome](https://www.runoob.com/images/compatible_chrome.gif)![Safari](https://www.runoob.com/images/compatible_safari.gif)

所有主要浏览器都支持 parseInt() 函数

------

## 提示和注释

**注意：** 只有字符串中的第一个数字会被返回。

**注意：** 开头和结尾的空格是允许的。

**注意：**如果字符串的第一个字符不能被转换为数字，那么 parseInt() 会返回 NaN。

**注意：**在字符串以"0"为开始时旧的浏览器默认使用八进制基数。ECMAScript 5，默认的是十进制的基数。

------

## 实例

## 实例

我们将使用 parseInt() 来解析不同的字符串：

```javascript
document.write(parseInt("10") + "<br>");
document.write(parseInt("10.33") + "<br>");
document.write(parseInt("34 45 66") + "<br>");
document.write(parseInt(" 60 ") + "<br>");
document.write(parseInt("40 years") + "<br>");
document.write(parseInt("He was 40") + "<br>");
 
document.write("<br>");
document.write(parseInt("10",10)+ "<br>");
document.write(parseInt("010")+ "<br>");
document.write(parseInt("10",8)+ "<br>");
document.write(parseInt("0x10")+ "<br>");
document.write(parseInt("10",16)+ "<br>");
```



以上实例输出结果：

```
10
10
34
60
40
NaN

10
10
8
16
16
```




[尝试一下 »](https://www.runoob.com/try/try.php?filename=tryjsref_parseint)

**注意：**旧浏览器由于使用旧版本的ECMAScript（ECMAScript版本小于ECMAScript 5，当字符串以"0"开头时默认使用八进制，ECMAScript 5使用的是十进制），所以在解析("010") 将输出8。