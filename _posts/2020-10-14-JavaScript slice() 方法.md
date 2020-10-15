---
layout:     post           # 使用的布局（不需要改）
title:      JavaScript slice() 方法           # 标题 
subtitle:   JavaScript slice() 方法 #副标题
date:       2020-10-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [JavaScript slice() 方法](https://juejin.im/post/6844903569087266823)

# JavaScript slice() 方法

[![Array 对象参考手册](https://www.runoob.com/images/up.gif) JavaScript Array 对象](https://www.runoob.com/jsref/jsref-obj-array.html)

## 实例

在数组中读取元素：

var fruits = ["Banana", "Orange", "Lemon", "Apple", "Mango"];
var citrus = fruits.slice(1,3);

*citrus* 结果输出:

Orange,Lemon


[尝试一下 »](https://www.runoob.com/try/try.php?filename=tryjsref_slice_array)

------

## 定义和用法

slice() 方法可从已有的数组中返回选定的元素。

slice()方法可提取字符串的某个部分，并以新的字符串返回被提取的部分。

**注意：** slice() 方法不会改变原始数组。

------

## 浏览器支持

![Internet Explorer](https://www.runoob.com/images/compatible_ie.gif)![Firefox](https://www.runoob.com/images/compatible_firefox.gif)![Opera](https://www.runoob.com/images/compatible_opera.gif)![Google Chrome](https://www.runoob.com/images/compatible_chrome.gif)![Safari](https://www.runoob.com/images/compatible_safari.gif)

所有主要浏览器都支持slice()。

------

## 语法

*array*.slice(*start*, *end*)

## 参数 Values

| 参数    | 描述                                                         |
| :------ | :----------------------------------------------------------- |
| *start* | 可选。规定从何处开始选取。如果是负数，那么它规定从数组尾部开始算起的位置。也就是说，-1 指最后一个元素，-2 指倒数第二个元素，以此类推。 |
| *end*   | 可选。规定从何处结束选取。该参数是数组片断结束处的数组下标。如果没有指定该参数，那么切分的数组包含从 start 到数组结束的所有元素。如果这个参数是负数，那么它规定的是从数组尾部开始算起的元素。 |

## 返回值

| Type  | 描述                                                         |
| :---- | :----------------------------------------------------------- |
| Array | 返回一个新的数组，包含从 start 到 end （不包括该元素）的 arrayObject 中的元素。 |

## 技术细节

| JavaScript 版本: | 1.2  |
| :--------------- | ---- |
|                  |      |

------

## 更多实例

## 实例

使用负值从数组中读取元素

```javascript
var fruits = ["Banana", "Orange", "Lemon", "Apple", "Mango"];
var myBest = fruits.slice(-3,-1);
```

*myBest* 结果输出:

```
Lemon,Apple
```


[尝试一下 »](https://www.runoob.com/try/try.php?filename=tryjsref_slice_array2)

## 实例

截取字符串

```javascript
var str="www.runoob.com!";
document.write(str.slice(4)+"<br>"); // 从第 5 个字符开始截取到末尾
document.write(str.slice(4,10)); // 从第 5 个字符开始截取到第10个字符
```

