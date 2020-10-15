---
layout:     post           # 使用的布局（不需要改）
title:      JavaScript substr() 方法           # 标题 
subtitle:   JavaScript substr() 方法 #副标题
date:       2020-10-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [JavaScript substr() 方法](https://www.runoob.com/jsref/jsref-substring.html)

# JavaScript substr() 方法

[![String 对象参考手册](https://www.runoob.com/images/up.gif) JavaScript String 对象](https://www.runoob.com/jsref/jsref-obj-string.html)

## 实例

抽取指定数目的字符：

var str="Hello world!";
var n=str.substr(2,3)

*n* 输出结果:

llo


[尝试一下 »](https://www.runoob.com/try/try.php?filename=tryjsref_substr)

------

## 定义和用法

substr() 方法可在字符串中抽取从 *开始* 下标开始的指定数目的字符。

**提示：** substr() 的参数指定的是子串的开始位置和长度，因此它可以替代 substring() 和 slice() 来使用。
在 IE 4 中，参数 start 的值无效。在这个 BUG 中，start 规定的是第 0 个字符的位置。在之后的版本中，此 BUG 已被修正。
ECMAscript 没有对该方法进行标准化，因此反对使用它。

**注意：** substr() 方法不会改变源字符串。

------

## 浏览器支持

![Internet Explorer](https://www.runoob.com/images/compatible_ie.gif)![Firefox](https://www.runoob.com/images/compatible_firefox.gif)![Opera](https://www.runoob.com/images/compatible_opera.gif)![Google Chrome](https://www.runoob.com/images/compatible_chrome.gif)![Safari](https://www.runoob.com/images/compatible_safari.gif)

所有主要浏览器都支持 substr() 方法

------

## 语法

*string*.substr(*start*,*length*)

## 参数值

| 参数     | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| *start*  | 必需。要抽取的子串的起始下标。必须是数值。如果是负数，那么该参数声明从字符串的尾部开始算起的位置。也就是说，-1 指字符串中最后一个字符，-2 指倒数第二个字符，以此类推。 |
| *length* | 可选。子串中的字符数。必须是数值。如果省略了该参数，那么返回从 stringObject 的开始位置到结尾的字串。 |

## 返回值

| 类型   | 描述                                                   |
| :----- | :----------------------------------------------------- |
| String | A new string containing the extracted part of the text |

## 技术细节

| JavaScript 版本： | 1.0  |
| :---------------- | ---- |
|                   |      |



------

## 更多实例

## 实例

在本例中，我们将使用 substr() 从字符串第二个位置中提取一些字符：

```javascript
var str="Hello world!";
var n=str.substr(2)
```



*n* 输出结果:

```javascript
llo world!
```

