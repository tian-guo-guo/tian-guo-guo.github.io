---
layout:     post           # 使用的布局（不需要改）
title:      splice slice           # 标题 
subtitle:   splice slice #副标题
date:       2020-10-12             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# splice slice

## 一、array.splice()

```javascript
var array = [1,2,3,4,5];
console.log(array.splice(1,2,3,4,5));
console.log(array);

[ 2, 3 ]
[ 1, 3, 4, 5, 4, 5 ]
```



Array数组的splice()方法，也是一个非常强大的方法，它的作用是:删除、插入、替换

需要注意的是： splice()方法是直接修改原数组的

一、删除的用法

语法: array.splice(starti,n);

starti 指的是从哪个位置开始(不包含starti)

n指的是需要删除的个数

```html
<script>
    var array=[1,2,3,4,5];
    array.splice(3,2);
    console.log(array);
</script>
```

结果： [1,2,3]

这里有个小拓展：其实被删除的元素可以用一个变量接收的，这个接收的变量可以作为拼接数组来使用

```html
<script>
    var array=[1,2,3,4,5];
    var deletes =array.splice(3,2);
    console.log(deletes);
    console.log(array);
</script>

```

结果： [4,5]   [1,2,3]

```
我们将删除后的元素在拼接回原来的数组
<script>
    var array=[1,2,3,4,5];
    var deletes =array.splice(3,2);
    console.log(deletes);
    console.log(array);
    array=array.concat(deletes);
    console.log(array);
</script>

```

结果：  [4,5]   [1,2,3]  [1,2,3,4,5]

## 二、插入的用法
语法：array.splice(starti,0,值1，值2...);
starti: 在哪个位置插入，原来starti位置的值向后顺移
0：表示删除0个元素，因为插入和替换都是由删除功能拓展的。
值1，值2：需要插入的值

```
<script>
    var array=[1,2,3,4,5];
    array.splice(2,0,123,456);
    console.log(array);
</script>
```


结果： [1,2,123,456,3,4,5]

## 三、替换的用法
语法:array.splice(starti,n,值1，值2);
原理和插入的用法相同
实际是就是：在starti的位置删除n个元素，然后在这个位置插入值1，值2,就可以起到替换
原来被删除的值

```html
<script>
    var array=[1,2,3,4,5];
    array.splice(2,2,123,456);
    console.log(array);
</script>
结果：[1,2,123,456,5]
```

总结:

splice()方法实际是一个删除数组元素方法，但可以拓展出插入，和替换两个用法

# slice()方法

```javascript
var array = [1,2,3,4,5];
console.log(array.slice(1,4));
console.log(array);

[ 2, 3, 4 ]
[ 1, 2, 3, 4, 5 ]
```



slice() 方法可从已有的数组中返回选定的元素。
 slice()方法可提取字符串的某个部分，并以新的字符串返回被提取的部分。
 注意： slice() 方法不会改变原始数组。

**Array.prototype.slice()**

slice() 方法返回一个从开始到结束（不包括结束）选择的数组的一部分浅拷贝到一个新数组对象。
 原始数组不会被修改。



```css
arrayObject.slice(start,end)
```

-   start 必需。规定从何处开始选取。如果是负数，那么它规定从数组尾部开始算起的位置。也就是说，-1 指最后一个元素，-2 指倒数第二个元素，以此类推。
-   end   可选。规定从何处结束选取。该参数是数组片断结束处的数组下标。如果没有指定该参数，那么切分的数组包含从 start 到数组结束的所有元素。如果这个参数是负数，那么它规定的是从数组尾部开始算起的元素。

返回一个新的数组，包含从 start 到 end （不包括该元素）的 arrayObject 中的元素。

slice不修改原数组，只会返回一个浅复制了原数组中的元素的一个新数组。原数组的元素会按照下述规则拷贝：