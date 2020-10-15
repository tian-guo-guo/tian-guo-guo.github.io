---
layout:     post           # 使用的布局（不需要改）
title:      JavaScript substring() 方法           # 标题 
subtitle:   JavaScript substring() 方法 #副标题
date:       2020-10-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [JavaScript substring() 方法](https://www.runoob.com/jsref/jsref-substring.html)

# JavaScript substring() 方法

[![String 对象参考手册](https://www.runoob.com/images/up.gif) JavaScript String 对象](https://www.runoob.com/jsref/jsref-obj-string.html)

------

## 定义和用法

substring() 方法用于提取字符串中介于两个指定下标之间的字符。

substring() 方法返回的子串包括 *开始* 处的字符，但不包括 *结束* 处的字符。

## 语法

*string*.substring(from, to)



| 参数 | 描述                                                         |
| :--- | :----------------------------------------------------------- |
| from | 必需。一个非负的整数，规定要提取的子串的第一个字符在 string Object 中的位置。 |
| to   | 可选。一个非负的整数，比要提取的子串的最后一个字符在 string Object 中的位置多 1。 如果省略该参数，那么返回的子串会一直到字符串的结尾。 |



------

## 浏览器支持

![Internet Explorer](https://www.runoob.com/images/compatible_ie.gif)![Firefox](https://www.runoob.com/images/compatible_firefox.gif)![Opera](https://www.runoob.com/images/compatible_opera.gif)![Google Chrome](https://www.runoob.com/images/compatible_chrome.gif)![Safari](https://www.runoob.com/images/compatible_safari.gif)

所有主要浏览器都支持 substring() 方法

------

## 实例

## 实例

在本例中，我们将使用 substring() 从字符串中提取一些字符：:

<script>  
    var str="Hello world!"; 
    document.write(str.substring(3)+"<br>"); 
    document.write(str.substring(3,7));  
</script>

以上代码输出结果:

```
lo world!
lo w
```

