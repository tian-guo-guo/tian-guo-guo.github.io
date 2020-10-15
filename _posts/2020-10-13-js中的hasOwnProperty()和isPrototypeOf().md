---
layout:     post           # 使用的布局（不需要改）
title:      js中的hasOwnProperty()和isPrototypeOf()           # 标题 
subtitle:   js中的hasOwnProperty()和isPrototypeOf() #副标题
date:       2020-10-13             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [js中的hasOwnProperty()和isPrototypeOf()](https://juejin.im/post/6844903569087266823)

# js中的hasOwnProperty()和isPrototypeOf()

>   这两个属性都是`Object.prototype`所提供:`Object.prototype.hasOwnProperty()`和`Object.prototype.isPropertyOf()`
>   先讲解`hasOwnProperty()`方法和使用。在讲解`isPropertyOf()`方法和使用

**看懂这些至少要懂原型链**

## 一、Object.prototype.hasOwnProperty()

### 概述

hasOwnProperty()方法用来判断某个对象是否含有指定的自身属性

### 语法

```
obj.hasOwnProperty("属性名");//实例obj是否包含有圆括号中的属性,是则返回true,否则是false复制代码
```

### 描述

所有继承了`Object.prototype`的对象都会从原型链上继承到`hasOwnProperty`方法，这个方法检测一个对象是否包含一个特定的属性，
和`in`不同，这个方法会忽略那些从原型链上继承的属性。

### 实例

#### 1.使用hasOwnProperty()方法判断某对象是否含有特定的自身属性

下面的例子检测了对象 o 是否含有自身属性 prop：

```
var o =new Object();
o.prop="exists";

function change(){
  o.newprop=o.prop;
  delete o.prop;
}

o.hasOwnProperty("prop")//true
change()//删除o的prop属性
o.hasOwnProperty("prop")//false
//删除后在使用hasOwnProperty()来判断是否存在，返回已不存在了复制代码
```

#### 2.自身属性和继承属性的区别

下面的列子演示了`hasOwnProperty()`方法对待自身属性和继承属性的区别。

```
var o =new Object();
o.prop="exists";
o.hasOwnProperty("prop");//true 自身的属性
o.hasOwnProperty("toString");//false 继承自Object原型上的方法
o.hasOwnProperty("hasOwnProperty");//false 继承自Object原型上的方法复制代码
```

#### 3.修改原型链后hasOwnProperty()的指向例子

下面的列子演示了`hasOwnProperty()`方法对待修改原型链后继承属性的区别

```
var o={name:'jim'};
function Person(){
  this.age=19;
}
Person.prototype=o;//修改Person的原型指向
p.hasOwnProperty("name");//false 无法判断继承的name属性
p.hasOwnProperty("age");//true;复制代码
```

#### 4.使用hasOwnProperty()遍历一个对象自身的属性

下面的列子演示了如何在遍历一个对象忽略掉继承属性，而得到自身属性。

**注意· `forin` 会遍历出对象继承中的可枚举属性**

```
var o={
  gender:'男'
}
function Person(){
  this.name="张三";
  this.age=19;
}
Person.prototype=o;
var p =new Person();
for(var k in p){
  if(p.hasOwnProperty(k)){
    console.log("自身属性："+k);// name ,age
  }else{
    console.log("继承别处的属性："+k);// gender
  }
}复制代码
```

#### 5.hasOwnProperty方法有可能会被覆盖

如果一个对象上拥有自己的`hasOwnProperty()`方法，则原型链上的`hasOwnProperty()`的方法会被覆盖掉

```
var o={
  gender:'男',
  hasOwnProperty:function(){
    return false;
  }
}

o.hasOwnProperty("gender");//不关写什么都会返回false
//解决方式，利用call方法
({}).hasOwnProperty.call(o,'gender');//true
Object.prototype.hasOwnProperty.call(o,'gender');//true复制代码
```

## 二、Object.prototype.isPrototypeOf()

### 概述

`isPrototypeOf()`方法测试一个对象是否存在另一个对象的原型链上

### 语法

```
//object1是不是Object2的原型,也就是说Object2是Object1的原型，,是则返回true,否则false
object1.isPrototypeOf(Object2);复制代码
```

### 描述

`isPrototypeOf()`方法允许你检查一个对像是否存在另一个对象的原型链上

### 实例

#### 1.利用isPrototypeOf()检查一个对象是否存在另一个对象的原型上

```
var o={};
function Person(){};
var p1 =new Person();//继承自原来的原型，但是现在已经无法访问
Person.prototype=o;
var p2 =new Person();//继承自o
console.log(o.isPrototypeOf(p1));//false o是不是p1的原型
console.log(o.isPrototypeof(p2));//true  o是不是p2的原型复制代码
```

#### 2.利用isPropertyOf()检查一个对象是否存在一另一个对象的原型链上

```
var o={};
function Person(){};
var p1 =new Person();//继承自原来的原型，但是现在已经无法访问
Person.prototype=o;
var p2 =new Person();//继承自o
console.log(o.isPrototypeOf(p1));//false o是不是p1的原型
console.log(o.isPrototypeof(p2));//true  o是不是p2的原型

console.log(Object.prototype.isPrototypeOf(p1));//true
console.log(Object.prototype.isPrototypeOf(p2));//true复制代码
```

>   `p1`的原型链结构是`p1`=>原来的`Person.prototype`=>`Object.prototype`=>`null`
>   `p2`的原型链结构是`p2`=> `o` =>`Object.prototype`=>`null`
>   `p1`和`p2`都拥有`Object.prototype`所以他们都在`Object.Prototype`的原型链上

## 三、总结

1.  hasOwnProperty：是用来判断一个对象是否有你给出名称的属性或对象。不过需要注意的是，此方法无法检查该对象的原型链中是否具有该属性，该属性必须是对象本身的一个成员。
2.  isPrototypeOf是用来判断要检查其原型链的对象是否存在于指定对象实例中，是则返回true，否则返回false。



[Link](https://juejin.im/post/6844903855474343950)

