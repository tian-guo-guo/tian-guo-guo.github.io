---
layout:     post           # 使用的布局（不需要改）
title:      Selenium WebDriver的工作原理           # 标题 
subtitle:   Selenium WebDriver的工作原理 #副标题
date:       2020-10-20             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [**Selenium WebDriver的工作原理**](https://www.ruanyifeng.com/blog/2016/06/dns.html)

先通过一个简单的类比说个好理解的，这个比喻是我从美版知乎Quora上看到的，觉得比较形象、好理解拿来用用。

**我们可以把WebDriver驱动浏览器类比成出租车司机开出租车。**

在开出租车时有三个角色：

**乘客：**他/她告诉出租车司机去哪里，大概怎么走

**出租车司机：**他按照乘客的要求来操控出租车

**出租车：**出租车按照司机的操控完成真正的行驶，把乘客送到目的地

![img](https://pic1.zhimg.com/80/v2-7427917f48f96192dec0d9fb53bea998_1440w.jpg)



在WebDriver中也有类似的三个角色：

**工程师写的自动化测试代码：**自动化测试代码发送请求给浏览器的驱动（比如火狐驱动、谷歌驱动）

**浏览器的驱动：**它来解析这些自动化测试的代码，解析后把它们发送给浏览器

**浏览器：**执行浏览器驱动发来的指令，并最终完成工程师想要的操作。



所以在这个类比中：

\1. 工程师写的自动化测试代码就相当于是乘客

\2. 浏览器的驱动就相当于是出租车司机

\3. 浏览器就相当于是出租车



## **下面再从技术上解释下WebDriver的工作原理：**



从技术上讲，也同样是上面的三个角色：

**1. WebDriver API**（基于Java、Python、C#等语言）

对于java语言来说，就是下载下来的selenium的Jar包，比如selenium-java-3.8.1.zip包，代表Selenium3.8.1的版本



**2. 浏览器的驱动**（browser driver）

每个浏览器都有自己的驱动，均以exe文件形式存在

比如谷歌的chromedriver.exe、火狐的geckodriver.exe、IE的IEDriverServer.exe



**3. 浏览器**

浏览器当然就是我们很熟悉的常用的各种浏览器。



**那在WebDriver脚本运行的时候，它们之间是如何通信的呢？为什么同一个browser driver即可以处理java语言的脚本，也可以处理python语言的脚本呢？**

让我们来看一下，一条Selenium脚本执行时后端都发生了哪些事情：

1.  对于每一条Selenium脚本，一个http请求会被创建并且发送给浏览器的驱动
2.  浏览器驱动中包含了一个HTTP Server，用来接收这些http请求
3.  HTTP Server接收到请求后根据请求来具体操控对应的浏览器
4.  浏览器执行具体的测试步骤
5.  浏览器将步骤执行结果返回给HTTP Server
6.  HTTP Server又将结果返回给Selenium的脚本，如果是错误的http代码我们就会在控制台看到对应的报错信息。

**为什么使用HTTP协议呢？**

因为HTTP协议是一个浏览器和Web服务器之间通信的标准协议，而几乎每一种编程语言都提供了丰富的http libraries，这样就可以方便的处理客户端Client和服务器Server之间的请求request及响应response，WebDriver的结构中就是典型的C/S结构，WebDriver API相当于是客户端，而小小的浏览器驱动才是服务器端。



**那为什么同一个浏览器驱动即可以处理Java语言的脚本，也可以处理Python语言的脚本呢？**

这就要提到WebDriver基于的协议：**JSON Wire protocol**。

关于WebDriver的协议也是面试的时候经常会问到的问题。

JSON Wire protocol是在http协议基础上，对http请求及响应的body部分的数据的进一步规范。

我们知道在HTTP请求及响应中常常包括以下几个部分：http请求方法、http请求及响应内容body、http响应状态码等。



常见的http请求方法：

GET：用来从服务器获取信息。比如获取网页的标题信息

POST：向服务器发送操作请求。比如findElement，Click等

http响应状态码：

在WebDriver中为了给用户以更明确的反馈信息，提供了更细化的http响应状态码，比如：

7： NoSuchElement

11：ElementNotVisible

200：Everything OK



现在到了最关键的http请求及响应的body部分了：

body部分主要传送具体的数据，在WebDriver中这些数据都是以JSON的形式存在并进行传送的，这就是**JSON Wire protocol**。

JSON是一种数据交换的格式，是对XML的升级与替代，下面是一个JSON文件的例子：

```html
  {
 
    "firstname": "Alex",
    "lastname": "Smith",
    "moble": "13300000000"
  }
```

下面的例子是WebDriver中在成功找到一个元素后JSON Wire Protocol的返回：

```text
{"status" : 0, "value" : {"element" : "123422"}}
```

所以在Client和Server之间，只要是基于JSON Wire Protocol来传递数据，就与具体的脚本语言无关了，这样同一个浏览器的驱动就即可以处理Java语言的脚本，也可以处理Python语言的脚本了。