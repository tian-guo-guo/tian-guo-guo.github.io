---
layout:     post           # 使用的布局（不需要改）
title:      javascript的setTimeout()用法总结，js的setTimeout()方法           # 标题 
subtitle:   javascript的setTimeout()用法总结，js的setTimeout()方法           #副标题
date:       2020-10-12             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [javascript的setTimeout()用法总结，js的setTimeout()方法](https://tech.meituan.com/2018/09/27/fe-security.html)

## 引子

js的setTimeout方法用处比较多，通常用在页面刷新了、延迟执行了等等。但是很多javascript新手对setTimeout的用法还是不是很了解。虽然我学习和应用javascript已经两年多了，但是对setTimeout方法，有时候也要查阅资料。今天对js的setTimeout方法做一个系统地总结。

## setInterval与setTimeout的区别

说道setTimeout，很容易就会想到setInterval，因为这两个用法差不多，但是又有区别，今天一起总结了吧！

### setTimeout

**定义和用法:** setTimeout()方法用于在指定的毫秒数后调用函数或计算表达式。　　

**语法:** setTimeout(code,millisec) 　

**参数：** code （必需）：要调用的函数后要执行的 JavaScript 代码串。millisec（必需）：在执行代码前需等待的毫秒数。 　 **提示：** setTimeout() 只执行 code 一次。如果要多次调用，请使用 setInterval() 或者让 code 自身再次调用 setTimeout()。

### setInterval

setInterval() 方法可按照指定的周期（以毫秒计）来调用函数或计算表达式。

setInterval() 方法会不停地调用函数，直到 clearInterval() 被调用或窗口被关闭。由 setInterval() 返回的 ID 值可用作 clearInterval() 方法的参数。

**语法:** setInterval(code,millisec[,"lang"])

**参数:** code 必需。要调用的函数或要执行的代码串。millisec 必须。周期性执行或调用 code 之间的时间间隔，以毫秒计。

**返回值:** 一个可以传递给 Window.clearInterval() 从而取消对 code 的周期性执行的值。

### 区别

通过上面可以看出，setTimeout和setinterval的最主要区别是：

setTimeout只运行一次，也就是说设定的时间到后就触发运行指定代码，运行完后即结束。如果运行的代码中再次运行同样的setTimeout命令，则可循环运行。（即 要循环运行，需函数自身再次调用 setTimeout()）

而 setinterval是循环运行的，即每到设定时间间隔就触发指定代码。这是真正的定时器。

setinterval使用简单，而setTimeout则比较灵活，可以随时退出循环，而且可以设置为按不固定的时间间隔来运行，比如第一次1秒，第二次2秒，第三次3秒。

我个人而言，更喜欢用setTimeout多一些！

## setTimeout的用法

**让我们一起来运行一个案例，首先打开记事本，将下面代码贴入，运行一下效果！**

```
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
</head>
<body>
<h1> <font color=blue> haorooms博客示范网页 </font> </h1>
<p> 请等三秒!</p>

<script>
setTimeout("alert('对不起, haorooms博客要你等候多时')", 3000 )
</script>

</body> 
</html>
```

页面会在停留三秒之后弹出对画框！这个案例应用了setTimeout最基本的语法，setTimeout不会自动重复执行！

**setTimeout也可以执行function，还可以不断重复执行！我们再来一起做一个案例：**

```
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<script>
var x = 0
function countSecond()
{
   x = x+1
　 document.haorooms.haoroomsinput.value=x
　 setTimeout("countSecond()", 1000)
}
</script>
</head>
<html>
<body>

<form name="haorooms">
   <input type="text" name="haoroomsinput"value="0" size=4 >
</form>

<script>
countSecond()
</script>

</body> </html>
```

你可以看到input文本框中的数字在一秒一秒的递增！所以，setTimeout也可以制作网页中的时间跳动！

没有案例，学习起来不会很快，我们再来一起做一个例子，计算你在haorooms某个页面的停留时间：

```
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<script>
x=0
y=-1
function countMin()
{　y=y+1
　 document.displayMin.displayBox.value=y
　 setTimeout("countMin()",60000)
}
function countSec()
{　x = x + 1
　 z =x % 60
　 document.displaySec.displayBox.value=z
　 setTimeout("countSec()", 1000)
}
</script> </head>
<body>
<table> <tr valign=top> <td> 你在haorooms博客中的停留时间是: </td>
<td> 
<form name=displayMin>
   <input type=text name=displayBox value=0 size=4 >
</form> 
</td>
<td> 分 </td>
<td> 
<form name=displaySec> </td>
<td> <input type=text name=displayBox value=0 size=4 >
</form>
 </td>
<td> 秒。</td> </tr>
 </table>
<script>
countMin()
countSec()
</script>
</body>
</html>
```

怎么样，通过上面的例子，对setTimeout（）的用法，相信你都了解了吧！

## clearTimeout( )

我们再来一起看一下 clearTimeout( )，

clearTimout( ) 有以下语法 : 　

clearTimeout(timeoutID)

要使用 clearTimeout( ), 我们设定 setTimeout( ) 时 , 要给予这 setTimout( ) 一个名称 , 这名称就是 timeoutID , 我们叫停时 , 就是用这 timeoutID 来叫停 , 这是一个自定义名称 , 但很多人就以 timeoutID 为名。

在下面的例子 , 设定两个 timeoutID, 分别命名为 meter1 及 meter2, 如下 :

timeoutID 　↓ meter1 = setTimeout(“count1( )”, 1000) meter2 = setTimeout(“count2( )”, 1000)

使用这 meter1 及 meter2 这些 timeoutID 名称 , 在设定 clearTimeout( ) 时 , 就可指定对哪一个 setTimeout( ) 有效 , 不会扰及另一个 setTimeout( ) 的操作。

**下面请看 clearTimeout()的案例**

```
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">

<script>
x = 0
y = 0
function count1()
{　x = x + 1
　 document.display1.box1.value = x
　 meter1=setTimeout("count1()", 1000)
}
function count2()
{　y = y + 1
　 document.display2.box2.value = y
　 meter2=setTimeout("count2()", 1000)
}
</script> </head>
<body> 
<p> </br>
<form name="display1">
    <input type="text" name="box1" value="0" size=4 >
    <input type=button value="停止计时" onClick="clearTimeout(meter1) " >
    <input type=button value="继续计时" onClick="count1() " >
</form>
<p>
<form name="display2">
    <input type="text" name="box2" value="0" size=4 >
    <input type=button value="停止计时" onClick="clearTimeout(meter2) " >
    <input type=button value="继续计时" onClick="count2() " >
</form>

<script>
    count1()
    count2()
</script>
</body>
</html>
```