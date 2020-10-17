---
layout:     post           # 使用的布局（不需要改）
title:      JavaScript初学者必看“箭头函数”           # 标题 
subtitle:   JavaScript初学者必看“箭头函数” #副标题
date:       2020-10-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [JavaScript初学者必看“箭头函数”](https://blog.fundebug.com/2017/05/25/arrow-function-for-beginner/)

**译者按:** 箭头函数看上去只是语法的变动，其实也影响了`this`的作用域。



-   原文: [JavaScript: Arrow Functions for Beginners](https://hackernoon.com/javascript-arrow-functions-for-beginners-926947fc0cdc)
-   译者: [Fundebug](https://www.fundebug.com/)

**本文采用意译，版权归原作者所有**

本文我们介绍箭头(arrow)函数的优点。

### 更简洁的语法

我们先来按常规语法定义函数：

```
function funcName(params) {
    return params + 2;
}
funcName(2);
// 4
```

该函数使用箭头函数可以使用仅仅一行代码搞定！

```
var funcName = params => params + 2;
funcName(2);
// 4
```

是不是很酷！虽然是一个极端简洁的例子，但是很好的表述了箭头函数在写代码时的优势。我们来深入了解箭头函数的语法：

```
parameters => {
    statements;
};
```

如果没有参数，那么可以进一步简化：

```
() => {
    statements;
};
```

如果只有一个参数，可以省略括号:

```
parameters => {
    statements;
};
```

如果返回值仅仅只有一个表达式(expression), 还可以省略大括号：

```
parameters => expression

// 等价于:
function (parameters){
  return expression;
}
```

现在你已经学会了箭头函数的语法，我们来实战一下。打开 Chrome 浏览器开发者控制台，输入：

```
var double = num => num * 2;
```

我们将变量`double`绑定到一个箭头函数，该函数有一个参数`num`, 返回 `num * 2`。 调用该函数：

```
double(2);
// 4

double(3);
// 6
```

**一行代码搞定 BUG 监控：[Fundebug](https://www.fundebug.com/)**

### 没有局部`this`的绑定

和一般的函数不同，箭头函数不会绑定`this`。 或者说箭头函数不会改变`this`本来的绑定。
我们用一个例子来说明：

```
function Counter() {
    this.num = 0;
}
var a = new Counter();
```

因为使用了关键字`new`构造，Counter()函数中的`this`绑定到一个新的对象，并且赋值给`a`。通过`console.log`打印`a.num`，会输出 0。

```
console.log(a.num);
// 0
```

如果我们想每过一秒将`a.num`的值加 1，该如何实现呢？可以使用`setInterval()`函数。

```
function Counter() {
    this.num = 0;
    this.timer = setInterval(function add() {
        this.num++;
        console.log(this.num);
    }, 1000);
}
```

我们来看一下输出结果：

```
var b = new Counter();
// NaN
// NaN
// NaN
// ...
```

你会发现，每隔一秒都会有一个`NaN`打印出来，而不是累加的数字。到底哪里错了呢？
首先使用如下语句停止`setInterval`函数的连续执行：

```
clearInterval(b.timer);
```

我们来尝试理解为什么出错：根据上一篇博客讲解的规则，首先函数`setInterval`没有被某个声明的对象调用，也没有使用`new`关键字，再之没有使用`bind`, `call`和`apply`。`setInterval`只是一个普通的函数。实际上`setInterval`里面的`this`绑定到全局对象的。我们可以通过将`this`打印出来验证这一点：

```
function Counter() {
    this.num = 0;
    this.timer = setInterval(function add() {
        console.log(this);
    }, 1000);
}
var b = new Counter();
```

你会发现，整个`window`对象被打印出来。 使用如下命令停止打印：

```
clearInterval(b.timer);
```

回到之前的函数，之所以打印`NaN`，是因为`this.num`绑定到`window`对象的`num`，而`window.num`未定义。

那么，我们如何解决这个问题呢？使用箭头函数！使用箭头函数就不会导致`this`被绑定到全局对象。

```
function Counter() {
    this.num = 0;
    this.timer = setInterval(() => {
        this.num++;
        console.log(this.num);
    }, 1000);
}
var b = new Counter();
// 1
// 2
// 3
// ...
```

通过`Counter`构造函数绑定的`this`将会被保留。在`setInterval`函数中，`this`依然指向我们新创建的`b`对象。

为了验证刚刚的说法，我们可以将 `Counter`函数中的`this`绑定到`that`, 然后在`setInterval`中判断`this`和`that`是否相同。

```
function Counter() {
    var that = this;
    this.timer = setInterval(() => {
        console.log(this === that);
    }, 1000);
}
var b = new Counter();
// true
// true
// ...
```

正如我们期望的，打印值每次都是`true`。最后，结束刷屏的打印：

```
clearInterval(b.timer);
```

### 总结

-   箭头函数写代码拥有更加简洁的语法；
-   不会绑定`this`。