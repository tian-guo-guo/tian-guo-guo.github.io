---
layout:     post           # 使用的布局（不需要改）
title:      图解javascript原型链           # 标题 
subtitle:   图解javascript原型链 #副标题
date:       2020-10-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [图解javascript原型链](https://juejin.im/post/6844903936365690894)

作者: [HerryLo](https://github.com/HerryLo)

[本文永久有效链接: https://github.com/AttemptWeb......](https://github.com/AttemptWeb/Record/issues/11)

原型链和原型对象是js的核心，js以原型链的形式，保证函数或对象中的方法、属性可以让向下传递，按照面向对象的说法，这就是继承。而js通过原型链才得以实现函数或对象的继承，那么下面我们就来聊一聊js中的原型链。**以下图居多，请放心食用**。

## prototype和contructor

**prototype指向函数的原型对象，这是一个显式原型属性，只有函数才拥有该属性**。**contructor**指向原型对象的构造函数。

```
// 可以思考一下的打印结果，它们分别指向谁
function Foo() {}

console.log(Foo.prototype)
console.log(Foo.prototype.constructor)
console.log(Foo.__proto__)
console.log(Foo.prototype.__proto__)复制代码
```

下面来看看各个构造函数与它自己原型对象之间的关系：

![img](https://user-gold-cdn.xitu.io/2019/9/6/16d04cd034743d31?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

## proto

每个对象都有`_proto_`，它是隐式原型属性，指向了创建该对象的构造函数原型。由于js中是没有类的概念，而为了实现继承，通过 `_proto_` 将对象和原型联系起来组成原型链，就可以让对象访问到不属于自己的属性。

### 函数和对象之间的关系

[![img](https://user-gold-cdn.xitu.io/2019/9/6/16d025974e61505e?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)](https://camo.githubusercontent.com/8b5b047a703f91fe9b5a0374a5d79164dfeb2dd1/68747470733a2f2f7777772e6469646968656e672e636f6d2f32303139303930352f313536373639383539383631382e6a7067)

Foo、Function和Object都是函数，它们的`_proto_`都指向`Function.prototype`。

### 原型对象之间的关系

![img](https://user-gold-cdn.xitu.io/2019/9/6/16d025afb49e4db1?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

它们的`_proto_`都指向了`Object.prototype`。js原型链最终指向的是Object原型对象

## _proto_原型链图

[![img](https://user-gold-cdn.xitu.io/2019/9/6/16d025974e4de114?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)](https://camo.githubusercontent.com/26b8dff410544185fda9510d34ea8a8fe6e90bfe/68747470733a2f2f7777772e6469646968656e672e636f6d2f32303139303930352f313536373639393338373339342e6a7067)

相信只要你看懂了上面的图表，那么你应该就已经理解了js的原型链了。

## 总结

-   Function 和 Object 是两个函数。
-   **proto** 将对象和原型连接起来组成了原型链。
-   所有的函数的 **proto** 都指向Function原型对象。
-   **js的原型链最终指向的是Object原型对象(Object.prototype)**（在这里我将null排除在外了）。

![img](https://user-gold-cdn.xitu.io/2019/9/6/16d04ccc5d03fbc7?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)