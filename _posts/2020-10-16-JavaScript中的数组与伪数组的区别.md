---
layout:     post           # 使用的布局（不需要改）
title:      JavaScript中的数组与伪数组的区别           # 标题 
subtitle:   JavaScript中的数组与伪数组的区别 #副标题
date:       2020-10-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [JavaScript中的数组与伪数组的区别](https://www.cnblogs.com/chenpingzhao/p/4764791.html)

在JavaScript中，除了5种原始数据类型之外，其他所有的都是对象，包括函数（Function）。

基本数据类型：String,boolean,Number,Undefined, Null

引用数据类型：Object(Array,Date,RegExp,Function)

在这个前提下，咱们再来讨论JavaScript的对象。

### 1、创建对象

```
var obj = {}; //种方式创建对象，被称之为对象直接量（Object Literal）
var obj = new Object(); // 创建一个空对象，和{}一样
```

更多创建对象的知识，参见《JavaScript权威指南（第6版）》第6章

**2、创建数组**

```
var arr = [];//这是使用数组直接量（Array Literal）创建数组
var arr = new Array();//构造函数Array() 创建数组对象
```

 更多创建数组的知识，参见《JavaScript权威指南（第6版）》第7章。

### 3、对象与数组的关系

在说区别之前，需要先提到另外一个知识，就是JavaScript的原型继承。所有JavaScript的内置构造函数都是继承自 Object.prototype。在这个前提下，可以理解为使用 new Array() 或 [] 创建出来的数组对象，都会拥有 Object.prototype 的属性值。

对于题主的问题意味着：

```
var obj = {};// 拥有Object.prototype的属性值
var arr = [];
//使用数组直接量创建的数组，由于Array.prototype的属性继承自 Object.prototype，
//那么，它将同时拥有Array.prototype和Object.prototype的属性值
```

可以得到对象和数组的第一个区别：对象没有数组Array.prototype的属性值

### 4、什么是数组

数组具有一个最基本特征：索引，这是对象所没有的，下面来看一段代码：

```
var obj = {};
var arr = [];
 
obj[2] = 'a';
arr[2] = 'a';
 
console.log(obj[2]); // 输出 a
console.log(arr[2]); // 输出 a
console.log(obj.length); // 输出 undefined
console.log(arr.length); // 输出 3
```

通过上面这个测试，可以看到，虽然 obj[2]与arr[2] 都输出'a'，但是，在输出length上有明显的差异，这是为什么呢？

**obj[2]与arr[2]的区别**

-   obj[2]输出'a'，是因为对象就是普通的键值对存取数据
-   而arr[2]输出'a' 则不同，数组是通过索引来存取数据，arr[2]之所以输出'a'，是因为数组arr索引2的位置已经存储了数据

**obj.length与arr.length的区别**

-   obj.length并不具有数组的特性，并且obj没有保存属性length，那么自然就会输出undefined
-   而对于数组来说，length是数组的一个内置属性，数组会根据索引长度来更改length的值。

**为什么arr.length输出3，而不是1呢？**

-   这是由于数组的特殊实现机制，对于普通的数组，如果它的索引是从0开始连续的，那么length的值就会等于数组中元素个数
-   而对于上面例子中arr，在给数组添加元素时，并没有按照连续的索引添加，所以导致数组的索引不连续，那么就导致索引长度大于元素个数，那么我们称之为稀疏数组。

有关稀疏数组的特性就不再讨论更多，参见《JavaScript权威指南（第6版）》7.3节。

### 5、伪数组

定义：

1、拥有length属性，其它属性（索引）为非负整数(对象中的索引会被当做字符串来处理，这里你可以当做是个非负整数串来理解)
2、不具有数组所具有的方法

伪数组，就是像数组一样有 `length` 属性，也有 `0`、`1`、`2`、`3` 等属性的对象，看起来就像数组一样，但不是数组，比如

```
var fakeArray = {
    length: 3,
    "0": "first",
    "1": "second",
    "2": "third"
};
 
for (var i = 0; i < fakeArray.length; i++) {
    console.log(fakeArray[i]);
}
 
Array.prototype.join.call(fakeArray,'+');
```

常见的参数的参数 arguments，DOM 对象列表（比如通过 document.getElementsByTags 得到的列表），jQuery 对象（比如 $("div")）。

伪数组是一个 Object，而真实的数组是一个 Array

```
fakeArray instanceof Array === false;
Object.prototype.toString.call(fakeArray) === "[object Object]";
 
var arr = [1,2,3,4,6];
arr instanceof Array === true;
Object.prototype.toString.call(arr) === "[object Array]"
```

《javascript权威指南》上给出了代码用来判断一个对象是否属于“类数组”。如下：

```
// Determine if o is an array-like object.
// Strings and functions have numeric length properties, but are
// excluded by the typeof test. In client-side JavaScript, DOM text
// nodes have a numeric length property, and may need to be excluded
// with an additional o.nodeType != 3 test.
function isArrayLike(o) {   
    if (o &&                                // o is not null, undefined, etc.
            typeof o === 'object' &&            // o is an object
            isFinite(o.length) &&               // o.length is a finite number
            o.length >= 0 &&                    // o.length is non-negative
            o.length===Math.floor(o.length) &&  // o.length is an integer
            o.length < 4294967296)              // o.length < 2^32
            return true;                        // Then o is array-like
    else
            return false;                       // Otherwise it is not
}
```

 不过有个更简单的办法来判断，用 `Array.isArray`

```
Array.isArray(fakeArray) === false;
Array.isArray(arr) === true;
```

 从外观上看伪数组，看不出来它与数组的区别，在JavaScript内置对象中常见的伪数组就是大名鼎鼎的auguments：

```
function() {
  console.log(typeof arguments); // 输出 object，它并不是一个数组
}());
```

 另外在DOM对象中，childNodes也是伪数组

```
console.log(typeof document.body.childNodes); // 输出 object
```

除此之外，还有很多常用的伪数组，就不一一列举。

伪数组存在的意义，是可以让普通的对象也能正常使用数组的很多算法，比如：

```
var arr = Array.prototype.slice.call(arguments)
 
或者
var arr = Array.prototype.slice.call(arguments, 0); // 将arguments对象转换成一个真正的数组
 
Array.prototype.forEach.call(arguments, function(v) {
  // 循环arguments对象
});
```

 除了使用 Array.prototype.slice.call(arguments)，你也可以简单的使用[].slice.call(arguments) 来代替。另外，你可以使用 bind 来简化该过程。

```
var unboundSlice = Array.prototype.slice;
var slice = Function.prototype.call.bind(unboundSlice);
 
function list() {
  return slice(arguments);
}
 
var list1 = list(1, 2, 3); // [1, 2, 3]
```

将具有length属性的对象转换成数组对象，arguments是每个函数在运行的时候自动获得的一个近似数组的对象（传入函数的参数从0开始按数字排列，而且有length）。

比如当你 func('a', 'b', 'c') 的时候，func里面获得的arguments[0] 是 'a'，arguments[1] 是 'b'**，依次类推。但问题在于这个arguments对象其实并不是Array，所以没有slice方法。Array.prototype.slice.call( )可以间接对其实现slice的效果，而且返回的结果是真正的Array**。

对于IE9以前的版本(DOM实现基于COM)，我们可以使用makeArray来实现。

```
// 伪数组转化成数组
var makeArray = function(obj) {   
    if (!obj || obj.length === 0) {       
        return [];
    }   
    // 非伪类对象，直接返回最好
    if (!obj.length) {       
        return obj;
    }   
    // 针对IE8以前 DOM的COM实现
    try {       
        return [].slice.call(obj);
    } catch (e) {       
        var i = 0,
            j = obj.length,
            res = [];       
        for (; i < j; i++) {
            res.push(obj[i]);
        }
        return res;
    }
 
};
```

 更多关于伪数组的知识，参见《JavaScript权威指南（第6版）》7.11节。

### 6、总结

-   对象没有数组Array.prototype的属性值，类型是Object，而数组类型是Array；
-   数组是基于索引的实现，length会自动更新，而对象是键值对；
-   使用对象可以创建伪数组，伪数组可以正常使用数组的大部分方法；