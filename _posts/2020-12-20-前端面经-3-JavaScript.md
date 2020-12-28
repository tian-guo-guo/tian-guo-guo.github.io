---
layout:     post           # 使用的布局（不需要改）
title:      前端面经总结
subtitle:   前端面经总结 #副标题
date:       2020-12-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试

---

## JavaScript

### 创建一个对象

```js
function Person(name, age) {
  this.name = name;
  this.age = age;
  this.sing = function () {
    alert(this.name)
  }
}
```

### 谈谈对this对象的理解？

this 是 js 的一个关键字，随着函数使用场合不同，this 的值会发生变化。

但是总有一个原则，那就是 this 指的是调用函数的那 个对象。

this一般情况下：是全局对象 Global。 作为方法调 用，那么 this 就是指这个对象

>全局作用域下的this指向window
>如果给元素的事件行为绑定函数，那么函数中的this指向当前被绑定的那个元素
>函数中的this，要看函数执行前有没有 . , 有 . 的话，点前面是谁，this就指向谁，如果没有点，指向window
>自执行函数中的this永远指向window
>定时器中函数的this指向window
>构造函数中的this指向当前的实例
>call、apply、bind可以改变函数的this指向
>箭头函数中没有this，如果输出this，就会输出箭头函数定义时所在的作用域中的this

this绑定函数的执行上下文，谁调用它，它就指向谁。分为默认绑定、显式绑定、隐式绑定、apply/call/bind绑定、new绑定和箭头函数绑定

默认绑定:严格模式下this指向undefined，非严格模式this指向window

call、apply、bind都可以改变this的指向，但是apply接收参数数组，call接收的是参数列表 bind接收的是参数列表，但是apply和call调用就执行，bind需要手动执行

箭头函数绑定:箭头函数的this是父作用域的this，不是调用时的this,其他方法的this是动态的，而箭头函数的this是静态的

```js
window.name='a'
const obj={
    name:'b',
    age:22,
    getName:()=>{
        console.log(this)
        console.log(this.name)
    },
    getAge:function(){
        setTimeout(()=>{
            console.log(this.age)
        })
    }
}
obj.getName();//window a
obj.getAge();//22
```

优先级:箭头函数>new绑定>显示绑定/apply/bind/call>隐式绑定>默认绑定

### New创建对象发生了什么，用代码写出来

```js
var obj={};
obj._proto_=Base.prototype;
Bash.call(obj);
```

1.  创建一个空对象，并且this变量引用该对象，同时还继承了这个函数的原型
2.  属性和方法被加入到this引用的对象里
3.  新创建的对象由this引用，最后隐式返回this

### JSON 的了解？

JSON(JavaScript Object Notation) 是一种轻量 级的数据交换格式。它是基于 JavaScript的一个子 集。数据格式简单, 易于读写, 占用带宽小 {'age':'12', 'name':'back'}

### js延迟加载的方式有哪些？

defer和 async、动态创建 DOM 方式（用得最多）、 按需异步载入 js

### ES5的继承和ES6的继承有什么区别？

ES5的继承时通过prototype或构造函数机制来实现。**ES5的继承实质上是先创建子类的实例对象，然后再将父类的方法添加到this上**（Parent.apply(this)）。

ES6的继承机制完全不同，**实质上是先创建父类的实例对象this（所以必须先调用父类的super()方法），然后再用子类的构造函数修改this**。

具体的：ES6通过class关键字定义类，里面有构造方法，类之间通过extends关键字实现继承。子类必须在constructor方法中调用super方法，否则新建实例报错。因为子类没有自己的this对象，而是继承了父类的this对象，然后对其进行加工。如果不调用super方法，子类得不到this对象。

ps：super关键字指代父类的实例，即父类的this对象。在子类构造函数中，调用super后，才可使用this关键字，否则报错。

### **es6新特性**

说的越多越好

**1. const 和 let**

`let`: 声明在代码块内有效的变量。

特点：

1.  在存在变理提升（不能在变量声明之前使用）
2.  let的暂时性死区： 其实与1差不多，只要在块作用域有声明，就不能在本作用域声明前用主个变量。
3.  不允许重复声明。

`const`: 声明一个只读的常量

特点：

1.  一但声明，这个值不能被改变（对于引用类型，是引用的地址不能被改变）
2.  声明时必须赋值

>   面试中常会问到var let const 三都的区别，回答的时候注重各自的特点，其实const let就是弥补var 的各种缺点，两都的特点就是var 的缺点。
>   工作中声明变量多用const 和 let
>   其中当声明的变量为引用类型如Array，如果没有直接更改引用的地址，可以使用const

**2. 解构赋值**

什么是解构赋值？

按照一定模式从数组或对象中提取值，然后对变量进行赋值（先提取，再赋值）

数组：

```js
let [a, b] = [1, 2]
// 以下的结果为右边数剩下的值所组成的数组
let [c, d ,...e] = [1, 2, 3, 4]
// 有默认值的写法
let [f = 100] = []  // f = 100
// 其中String也被视为类数组
let [a, b] = 'abcd' // a = a; b = b
```

对象:

变理名要与对象的属性名一样才可以：

```text
let { foo } = { foo: 1, bar: 2 } // foo = 1
// 重新命名（后面那个才是新变量）
let { foo: newFoo } = { foo: 1, bar: 2 } // newFoo = 1
```

实际使用：

1.  交换两个变量的值

```js
[x, y] = [y, x]
```

\2. 函数的封装

```js
function fn({ x, y } = {}) {
console.log(x, y)
}
```

其中，函数参数为一个对象，不会像`(x, y)`的形式这样固定参数的顺序，而`{} = {}`后面又赋一个空的对象就是为了在调用fn时不传参数而不会抛出错误导至程序中止

\3. 函数返回值的解构

函数返回多个值

```js
// 有次序的
function foo() {
return [a, b, c]
}
const [a, b, c] = foo()
// 无次序的
function foo() {
return { a, b, c}
}
const { b, a, c} = foo()
```

**3. 模板字符串**

```js
const h = 'hello'
`${ h } word`
```

`${}`中可以使用任意的javaScript表达试、运算、引用对象属性、函数调用等。结果是其返回值。

可以换行，但是所有的空格和换行会被保留。

特点：

可以换行，但是所有的空格和换行会被保留。

`${}`中可以使用任意的javaScript表达试、运算、引用对象属性、函数调用等。结果是其返回值。

**4. 函数的扩展**

1.  函数的默认值

```js
function m(x = 1) {}
```

\2. rest参数（用于获取函数的多余参数）

```js
function a(...values) {
// value 是一个数组，第个元素是传入的各个参数
}
```

\3. 函头函数

特点：

1.  函数体内的this = 定义时所在的对像
2.  不可以当作构造函数（不能用new)
3.  不可以用arguments对像，可以用rest
4.  不可以用yield命令（不能用作Generator函数）

>   阮老师的书中这一章讲到了有关尾调用，尾递归的内容，值得一看。

**5. 数组的扩展**

1.  扩展运算符。
2.  用于替代数组的`apply`。

call apply bind的区别：
用于改变this的指向， 第一个参数为this指向的对像，后面的参数是作为函数的参数。
区加在于：call apply 会即调用，而bind是生成一个等调用的函数。call bind参数是一个个用逗号罗列，而apply 是传入一个数组。

```js
fn.apply(null, [1, 2, 3])
fn.call(null, 1, 2, 3)
fn.bind(null, 1, 2, 3)()
// 指最大值
Math.max(...[3,4,5,62,8])
```

1.  合并数组

```text
// ES5
[1, 2].concat(3)
// ES6
[1, 2, ...[3]]
```

1.  新增的方法
2.  Array.from()将类数组转为数组

-   可遍历的对象(iterable)(Set, Map)
-   类似数组的对

```text
{ '0': 'a', '1': 'b' }
```

1.  实例的方法

-   `find()``findIndex()`找出第一个符合条件的成页/下标（位置）
-   `entries()``keys()``values()` 用于遍历数组。（配合for...of)
-   `includes()` 是否存在指定无素(返回布尔值)

**5. 对象的扩展**

1.  属性的简写：

```js
let a = 100
{ a }
// 等同于
{ a: 100 }
```

方法名同样可以简写，vue中就常常用这种写法：

```js
export default {
name: 'VueComp',
data() {},
create() {},
}
// 等同于
export default {
name: 'VueComp',
data: function() {},
create: function() {},
}
```

\2. 属性名可心使用表达式：

```js
let foo = 'foo'
let obj = {
[foo]: 'fooValue'
}
```

\3. 新增一些方法：

-   Object.is()
-   Object.assign()
-   对像的一些遍历：

Object.keys(), Object.values(), Object.entries()

```js
for(let key of Object.keys(obj)) {}
for(let value of Object.values(obj)) {}
for(let [key,value] of Object.entries(obj)){}
```

-   扩展运算符（常用）(es2017新增，在webpack中要另外的babel转换)

**6. Symbol**

javascript又新增的一种数据类型（第七种，另外6种为：`Undefined`、`Null`、`Boolean`、`String`、`Number`、`Object`)

注：symbol作为对象的属性名时不会被`for...in`,`for...of`,`Object.keys()`识别；可以改用`Reflect.ownkeys`方法.

**7. Set、Map**

Set和map是ES6新增的数据结构。

-   Set

特点： 1. 类似数组，但其成员是唯一的。

1.  是一个构造函数。

用法：

 数组去重：

```js
[...new Set([1,1,1,2,3,4,3])]
Array.from(new Set([1,1,1,2,3,4,3]))
```

-   Map

特点：

1.  为了解决javascript的对象只能用了符串作为键的问题。

用法： （使用实例的set,get,delete方法增，查，删）

```js
const m = new Map()
const obj = {a: 'aa'}
m.set(obj, 'obj as key')
m.get(obj) // 'obj as key'
m.delete(obj)
```

也可以在new 时接受一个数组

```js
const obj = { a: 'aa'}
const m = new Map([
['name': 'ym'],
[obj, { b: 'bbbb' }]
])
```

>   这段时间有一个很火的文章讲如何使用map组构来优化长长的if..else的

**8. Promise**

是异步编程的一种解决方案。

特点：

1.  状态不受外界影响（有三种状态：padding, fulfilled,redected)
2.  一旦状态改变就不会再变。

用法：

```js
const p = new Promise((resolve, reject) => {
setTimeout(() => {
resolve()
}, 1000)
}).then(res => {})
.catch(err => {})
```

注： then方法有两个参数，第一个是成功的回调，第二个为失败的回调，即：

```js
.then(res =>{}, err => {})
```

但是最好用catch方法， 因为catch方法会捕获then里的错误，then里有错误程序不会中止。

**Promise.all()**

将一组promise再包装成一个promise

```js
var pa = Promise.all([p1, p2, p3])
```

特点：

1.  当所有都fulfilledj时，promise.all才fulfilled
2.  当只有一个rejected时，promise.all就会rejected

**Iterator和for...of**

Iterator的3个作用：

1.  为各种数据提供统一的，简便的访问接口
2.  使数据结构的成员能按某种次序排列
3.  主要供for...of用

原生有iterator的数据结构：

```
Array`, `Map`, `Set`, `String`, `TypeArray`, `arguments`， `NodeList
```

(object是没有的)

**for...of与其他循环的比较**

1.  for循环写法比较麻烦
2.  数组的forEach: 无法用break;return跳出循环。
3.  For...in

-   数组的键名是数字，而for...in以字符串作为键名（也就是用for...in循环数组，其键名是字符串，笔者被坑过）
-   不仅可以遍历键名，还可以遍历手动添加的其他键，包括原型链上的
-   某些情况下，会心任意次序遍历
-   （ for...in主要为对象而设计的）

**9. Generator与async await**

generator是ES6提供的一种异步编程解决方案。使异步写法更像同步。

Async await是ES2017的标准，是generator的一个语法糖。

用法：

```js
async function a() {
await ...
console.log(111)
await ...
}
```

当执行a时，其不会阻塞涵数外面的代码（a内的代码会安顺序执行）

```js
console.log('开始')
a()
console.log('a后面')
// 开始 -> a后面 -> 111
```

**10. Class**

产生的原因： 原ES5语法的没有成型的类的概念。而面向对象编程又离不开类的概念。

ES5定义一个类:

```js
function Point(x, y) {
this.x = x;
this.y = y;
}
  
var p = new Point(1, 2)
```

ES6的class:

```js
class Point {
constructor(x, y) {
this.x = x;
this.y = y;
}
}
```

其中：

1.  constructor方法是类的默认方法，通过new 命令生成对象时会调用该方法，如果声明类时没有定义constructor，会默认定义一个空的。
2.  生成实例时必须用new ,不用会报错
3.  不存在变里提升（选定义类，再new实例）

**类的静态方法：**

所有在类中定义的方法都会被实例继承，如果不想被继承，可以在定义时加上static。表示为静态方法。

```js
class Foo {
static match() {}
}
Foo.match()
const f = new Foo()
f.match() // 报错
```

**类的静态属性**

很遗憾，ES6没有给类设静态属性，但是可以用以下方法定义(有提案，写方同静态方法)

```js
class Foo {}
Foo.porp = 1
// 使用
Foo.porp // 1
```

**类的实例属性**

类的方法默认被实例继承，那么属性呢？也是继承的，写法如下：

```js
class Foo {
myProp = 111;
...
}
```

**classr的继承 extends**

```js
class Point {}
class PointSon extends Point {
constructor(x, y, z) {
super(x, y)
this.z = z
}
}
```

其中：

1.  super等同于父类的constructor。
2.  子类必须在constructor中调用super， 也就是说用extends去继承一个类，就必须调用这个类（父类）的constructor。是因为子类没有自己的this对象，而是继承父类的this，然后对其进行加工
3.  如果了类没有写constructor，会默认生成一个，并包含了super(...args)

**11. Module**

一种将程序拆分成一个个小模块的支持，或者说是可以将一个个小模块加入到程序中去。

在ES6的module之前，比较流行的模块加载方案有:CommonJS和AMD，前者用于服务器（node)，后者用于浏览器。

区别：

1.  CommondJS和AMD是运行时加载的。
2.  module是编译时加载的。
3.  CommondJS输出的是值的复制，而ES6输出的是值的引用

**ES6模块默认使用严格模式**：

-   变里必须声明后再使用
-   函数的参数不能有同名属性
-   不能使用width
-   禁止this指向全局对象

**使用**

命令有： `export`、`import` 、`export default`

文件a.js

```js
export a = 1
export b = 2
```

相当于

```js
const a = 1;
const b = 2;
export { a, b }
```

在文件b.js中引入

```js
import { a, b } from './a.js'
```

引入是重命名

```js
import { a as reA, b as reB } from './a.js' // reA reB是重命名的变量
```

整体引入：

```js
import * as all from './a.js'
all.a // 1
all.b // 2
// all 相当于{ a, b }
```

**export default默认输出**

export default导出的模块在引入时可以自定义命名

```js
export default function() {
...
}
```

依然用import 引入,但是不用{}，且可以自定义变量名

```js
import name from './a.js'
name()
```

**从一个模块导入，然后再导出**

```js
// 写法一：
import { a, b } from './a.js'
export { a, b }
// 写法二：
export { a, b } from './a.js'
// 改名导出
export { a as reA, b } from './a.js'
// 整体导出
export * from './a.js'
```

**在浏览器中使用module**

将script标签的type设为module即可

```html
<!-- 方法一 -->
<script type="module" src="./a.js"></script>
<!-- 方法二 -->
<script type="module">
import { a } from './a.js'
</script>
```

其中：

-   type="module"的script内写的代码是在当前作用域，不是在全局。
-   模块内自动采用严格模式
-   顶层的this指向undefined
-   同一个模块如棵加载多次，只执行一次

### es6的class的es5的类有什么区别

```js
1.es6 class内部定义的方法都是不可枚举的
2.es6 class必须用new调用
3.es6 class不存在变量提升
4.es6 class默认使用严格模式
5.es6 class子类必须在父类的构造函数中调用super(),才有this对象；而es5是先有子类的this，再调用
```

### **`==和===区别是什么？`**

>   =赋值
>
>   ==返回一个布尔值；相等返回true，不相等返回false； 允许不同数据类型之间的比较； 如果是不同类型的数据进行，会默认进行数据类型之间的转换； 如果是对象数据类型的比较，比较的是空间地址
>
>   === 只要数据类型不一样，就返回false；

### **移动端的兼容问题**

>   给移动端添加点击事件会有300S的延迟 如果用点击事件，需要引一个fastclick.js文件，解决300s的延迟 一般在移动端用ontouchstart、ontouchmove、ontouchend
>   移动端点透问题,touchstart 早于 touchend 早于click,click的触发是有延迟的，这个时间大概在300ms左右，也就是说我们tap触发之后蒙层隐藏， 此时 click还没有触发，300ms之后由于蒙层隐藏，我们的click触发到了下面的a链接上 尽量都使用touch事件来替换click事件。例如用touchend事件(推荐)。 用fastclick，[github.com/ftlabs/fast…](https://link.zhihu.com/?target=https%3A//github.com/ftlabs/fastclick) 用preventDefault阻止a标签的click 消除 IE10 里面的那个叉号 input:-ms-clear{display:none;}
>   设置缓存 手机页面通常在第一次加载后会进行缓存，然后每次刷新会使用缓存而不是去重新向服务器发送请求。如果不希望使用缓存可以设置no-cache。
>
>   圆角BUG 某些Android手机圆角失效 background-clip: padding-box; 防止手机中网页放大和缩小 这点是最基本的，做为手机网站开发者来说应该都知道的，就是设置meta中的viewport
>
>   设置用户截止缩放，一般写视口的时候就已经写好了。

### **typeof和instance of 检测数据类型有什么区别？**

>   相同点： 都常用来判断一个变量是否为空，或者是什么类型的。
>
>   不同点： typeof 返回值是一个字符串，用来说明变量的数据类型 instanceof 用于判断一个变量是否属于某个对象的实例.

### **如何判断一个变量是对象还是数组（prototype.toString.call()）。**

```text
千万不要使用typeof来判断对象和数组，因为这种类型都会返回object。
```

>   typeOf()是判断基本类型的Boolean,Number，symbol, undefined, String。 对于引用类型：除function，都返回object null返回object。
>
>   installOf() 用来判断A是否是B的实例，installof检查的是原型。
>
>   toString() 是Object的原型方法，对于 Object 对象，直接调用 toString() 就能返回 [Object Object] 。而对于其他对象，则需要通过 call / apply 来调用才能返回正确的类型信息。
>
>   hasOwnProperty()方法返回一个布尔值，指示对象自身属性中是否具有指定的属性，该方法会忽略掉那些从原型链上继承到的属性。
>
>   isProperty()方法测试一个对象是否存在另一个对象的原型链上。

### **使元素消失的方法**

```js
visibility:hidden、display:none、z-index=-1、opacity：0
1.opacity：0,该元素隐藏起来了，但不会改变页面布局，并且，如果该元素已经绑定了一些事件，如click事件也能触发
2.visibility:hidden,该元素隐藏起来了，但不会改变页面布局，但是不会触发该元素已经绑定的事件
3.display:node, 把元素隐藏起来，并且会改变页面布局，可以理解成在页面中把该元素删掉
```

### 实现页面加载进度条

### **常见的设计模式有哪些？**

```text
1、js工厂模式
2、js构造函数模式
3、js原型模式
4、构造函数+原型的js混合模式
5、构造函数+原型的动态原型模式
6、观察者模式
7、发布订阅模式
```

### 事件委托

### 实现extend函数

### 为什么会有跨域的问题以及解决方式？

### jsonp原理、postMessage原理？

### 实现拖拽功能，比如把5个兄弟节点中的最后一个节点拖拽到节点1和节点2之间

### 动画：setTimeout何时执行，requestAnimationFrame的优点

### 手写parseInt的实现：要求简单一些，把字符串型的数字转化为真正的数字即可，但不能使用JS原生的字符串转数字的API，比如Number()

### 写一个通用的事件侦听器函数

```js
// event(事件)工具集，来源： https://github.com/markyun markyun.Event = {

// 页面加载完成后 readyEvent : function(fn) { if (fn==null) { fn=document; } var oldonload = window.onload; if (typeof window.onload != 'function') { window.onload = fn; } else { window.onload = function() { oldonload(); fn(); };
}

}, // 视能力分别使用 dom0||dom2||IE方式 来 绑定事件 // 参数： 操作的元素,事件名称 ,事件处理程序 addEvent : function(element, type, handler) { if (element.addEventListener) { //事件类型、需要执行的函数、是否捕捉 element.addEventListener(type, handler, false); } else if (element.attachEvent) { element.attachEvent('on' + type, function() { handler.call(element); }); } else { element['on' + type] = handler; }

}, // 移除事件 removeEvent : function(element, type,handler) { if (element.removeEnentListener) { element.removeEnentListener(type, handler, false); } else if (element.datachEvent) { element.detachEvent('on' + type, handler); } else { element['on' + type] = null; } }, // 阻止事件 (主要是事件冒泡，因为 IE不支持 事件捕获) stopPropagation : function(ev) { if (ev.stopPropagation) { ev.stopPropagation(); } else { ev.cancelBubble = true; }

}, // 取消事件的默认行为 preventDefault : function(event) {if (event.preventDefault) { event.preventDefault(); } else { event.returnValue = false; }

}, // 获取事件目标 getTarget : function(event) { return event.target || event.srcElement; }, // 获取 event对象的引用，取到事件的所有信 息，确保随时能使用 event； getEvent : function(e) { var ev = e || window.event; if (!ev) { var c = this.getEvent.caller; while (c) { ev = c.arguments[0]; if (ev && Event == ev.constructor) { break; }c = c.caller;}} return ev;}};
```



### 异步加载的方式

(1) defer， 只支持 IE

(2) async：

(3) 创建 script， 插入到 DOM中， 加载完毕 后 callBack

documen.write和 innerHTML的区别

document.write只能重绘整个页面

innerHTML可以重绘页面的一部分

### **async await函数**

>   async/await函数是异步代码的新方式
>
>   async/await是基于promise实现的
>
>   async/await使异步代码更像同步代码
>
>   await 只能在async函数中使用，不能再普通函数中使用，要成对出现
>
>   默认返回一个promise实例，不能被改变
>
>   await下面的代码是异步，后面的代码是同步的

### **JS中同步和异步,以及js的事件流**

>   同步：在同一时间内做一件事情
>
>   异步：在同一时间内做多个事情 JS是单线程的，每次只能做一件事情，JS运行在浏览器中，浏览器是多线程的，可以在同一时间执行多个任务。

### JS中常见的异步任务
定时器、ajax、事件绑定、回调函数、async await、promise

### 22.TCP的三次握手和四次挥手
三次握手

>   
>第一次握手：客户端发送一个SYN码给服务器，要求建立数据连接；
>   第二次握手： 服务器SYN和自己处理一个SYN（标志）；叫SYN+ACK（确认包）；发送给客户端，可以建立连接
>第三次握手： 客户端再次发送ACK向服务器，服务器验证ACK没有问题，则建立起连接；
>   四次挥手
>   
>第一次挥手： 客户端发送FIN(结束)报文，通知服务器数据已经传输完毕；
>   第二次挥手: 服务器接收到之后，通知客户端我收到了SYN,发送ACK(确认)给客户端，数据还没有传输完成
>   第三次挥手： 服务器已经传输完毕，再次发送FIN通知客户端，数据已经传输完毕
>第四次挥手： 客户端再次发送ACK,进入TIME_WAIT状态；服务器和客户端关闭连接；

### 为什么建立连接是三次握手，而断开连接是四次挥手呢?
建立连接的时候， 服务器在LISTEN状态下，收到建立连接请求的SYN报文后，把ACK和SYN放在一个报文里发送给客户端。 而关闭连接时，服务器收到对方的FIN报文时，仅仅表示对方不再发送数据了但是还能接收数据，而自己也未必全部数据都发送给对方了，所以己方可以立即关闭，也可以发送一些数据给对方后，再发送FIN报文给对方来表示同意现在关闭连接，因此，己方ACK和FIN一般都会分开发送，从而导致多了一次。

### 事件流

DOM2事件流分为三个部分:事件捕获、处于目标、事件冒泡。

**事件冒泡**是指事件从执行的元素开始往上层遍历执行

**事件捕获**是指事件从根元素开始从外向里执行

```js
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<button id="btn">click Me</button>  
<script>
let btn=document.getElementById('btn');
btn.onclick=fucntion(e){
    console.log(e)
}
</script>
</body>
</html>
```

点击按钮后，事件冒泡的执行顺序是:button->body->html->document

事件捕获的执行顺序则相反:document->html->body->button

### **前端事件流**

```text
事件流描述的是从页面中接受事件的顺序，事件 捕获阶段 处于目标阶段 事件冒泡阶段 addeventListener 最后这个布尔值参数如果是true，表示在捕获阶段调用事件处理程序；如果是false，表示在冒泡阶段调用事件处理程序。
  1、事件捕获阶段：实际目标div在捕获阶段不会接受事件，也就是在捕获阶段，事件从document到<html>再到<body>就停止了。
      2、处于目标阶段：事件在div发生并处理，但是事件处理会被看成是冒泡阶段的一部分。
      3、冒泡阶段：事件又传播回文档
   阻止冒泡事件event.stopPropagation()
	  function stopBubble(e) {
    		if (e && e.stopPropagation) { // 如果提供了事件对象event 这说明不是IE浏览器
      		e.stopPropagation()
    		} else {
      		window.event.cancelBubble = true //IE方式阻止冒泡
    	      }
  		   }
   阻止默认行为event.preventDefault()
 function stopDefault(e) {
    if (e && e.preventDefault) {
      e.preventDefault()
    } else {
      // IE浏览器阻止函数器默认动作的行为
      window.event.returnValue = false
    }
  }
```

### **事件如何先捕获后冒泡？**

>   在DOM标准事件模型中，是先捕获后冒泡。但是如果要实现先冒泡后捕获的效果， 对于同一个事件，监听捕获和冒泡，分别对应相应的处理函数，监听到捕获事件，先暂缓执行，直到冒泡事件被捕获后再执行捕获事件。
>
>   哪些事件不支持冒泡事件：鼠标事件：mouserleave mouseenter 焦点事件：blur focus UI事件：scroll resize

### **target和currentTarget区别**

target是事件的真正目标

currentTarget是事件处理程序注册的元素

**document.ready和window.onload区别**

document.ready是dom树加载后执行，而window.onload是整个页面资源加载完后执行，所以document.ready比window.onload先执行

### **setTimeout、setInterval区别**

两者都是定时器，设定一个150ms后执行的定时器不代表150ms后定时器会执行，它表示代码在150ms内会被加入队列，如果这个时间点队列没有其他逻辑在执行，表面上看代码在精确时间执行了。在队列中有其他逻辑时，代码等待时间会超过150ms

**setTimeout** 只执行一次

**setInterval** 执行多次，属于重复定时器

```text
因为js是单线程的。浏览器遇到etTimeout 和 setInterval会先执行完当前的代码块，在此之前会把定时器推入浏览器的
待执行时间队列里面，等到浏览器执行完当前代码之后会看下事件队列里有没有任务，有的话才执行定时器里的代码
```

### 从浏览器返回html到渲染出页面，再到中间涉及到的优化点？**前端有哪些页面优化方法?**

>   减少 HTTP请求数
>   从设计实现层面简化页面
>   合理设置 HTTP缓存
>   资源合并与压缩
>   合并 CSS图片，减少请求数的又一个好办法。
>   将外部脚本置底（将脚本内容在页面信息内容加载后再加载）
>   多图片网页使用图片懒加载。
>   在js中尽量减少闭包的使用
>   尽量合并css和js文件
>   尽量使用字体图标或者SVG图标，来代替传统的PNG等格式的图片
>   减少对DOM的操作
>   在JS中避免“嵌套循环”和 “死循环”
>   尽可能使用事件委托（事件代理）来处理事件绑定的操作

（看雅虎 14 条性能优化原则）

（1） 减少 http 请求次数：CSS Sprites, JS、CSS 源码压缩、图片大小控制合适；网页 Gzip，CDN 托 管，data缓存 ，图片服务器。

（2） 前端模板 JS+数据，减少由于 HTML 标签导 致的带宽浪费，前端用变量保存 AJAX 请求结果，每 次操作本地变量，不用请求，减少请求次数

（3） 用 innerHTML 代替 DOM操作， 减少 DOM 操作次数，优化 javascript 性能。

（4） 当需要设置的样式很多时设置 className 而不是直接操作 style。

（5） 少用全局变量、缓存 DOM 节点查找的结果。 减少 IO 读取操作。

（6） 避免使用 CSS Expression（css 表达式)又 称 Dynamic properties(动态属性)。

（7） 图片预加载，将样式表放在顶部，将脚本放 在底部 加上时间戳。

（8） 避免在页面的主体布局中使用 table，table 要等其中的内容完全下载之后才会显示出来，显示比 div+css 布局慢。

### **浏览器渲染原理及流程 DOM -> CSSOM -> render -> layout -> print**

```text
流程：解析html以及构建dom树 -> 构建render树 ->  布局render树 -> 绘制render树
概念：1.构建DOM树： 渲染引擎解析HTML文档，首先将标签转换成DOM树中的DOM node(包括js生成的标签)生成内容树
      2.构建渲染树： 解析对应的css样式文件信息（包括js生成的样式和外部的css）
      3.布局渲染树：从根节点递归调用，计算每一个元素的大小，位置等。给出每个节点所在的屏幕的精准位置
      4.绘制渲染树：遍历渲染树，使用UI后端层来绘制每一个节点

重绘：当盒子的位置、大小以及其他属性，例如颜色、字体大小等到确定下来之后，浏览器便把这些颜色都按照各自的特性绘制一遍，将内容呈现在页面上
	触发重绘的条件：改变元素外观属性。如：color，background-color等
	重绘是指一个元素外观的改变所触发的浏览器行为，浏览器会根据元素的新属性重新绘制，使元素呈现新的外观
注意：table及其内部元素需要多次计算才能确定好其在渲染树中节点的属性值，比同等元素要多发时间，要尽量避免使用table布局

重排（重构/回流/reflow）： 当渲染书中的一部分（或全部）因为元素的规模尺寸，布局，隐藏等改变而需要重新构建，这就是回流。
	每个页面都需要一次回流，就是页面第一次渲染的时候

重排一定会影响重绘，但是重绘不一定会影响重排
```

>   将html代码按照深度优先遍历来生成DOM树。 css文件下载完后也会进行渲染，生成相应的CSSOM。 当所有的css文件下载完且所有的CSSOM构建结束后，就会和DOM一起生成Render Tree。 接下来，浏览器就会进入Layout环节，将所有的节点位置计算出来。 最后，通过Painting环节将所有的节点内容呈现到屏幕上。



### DOM和css如何解析，如何渲染出元素？

### **回流和重绘区别**

回流：当渲染树中元素尺寸、结构或者某些属性发生变化时，浏览器重新渲染部分或全部页面的情况叫回流。下列元素改变引发回流:

-   getBoundingClientRect()
-   scrollTo()
-   scrollIntoView()或者scrollIntoViewIfneeded
-   clientTop、clientLeft、clientWidth、clientHeight
-   offsetTop、offsetLeft、offsetWidth、offsetHeight
-   scrollTop、scrollLeft、scrollWidth、scrollHeight
-   getComputedStyle()

重绘：当页面中元素样式变化不会改变它在文档流中的位置时，即不会使元素的几何属性发生变化，浏览器会将新样式赋给它并重新绘制页面(比如color、backgroundColor)

>   频繁回流和重绘会引起性能问题

避免方法:

-   减少table布局使用
-   减少css表达式的使用(如calc())
-   减少DOM操作，用documentFragment代替
-   将元素设为display:none;操作结束后把它显示回来，因为display:none不会引发回流重绘
-   避免频繁读取会引发回流重绘的元素，如果需要最好是缓存起来
-   对复杂动画元素使用绝对定位，使它脱离文档流
-   减少使用行内样式

### **防抖节流**

节流:多次触发事件时，一段时间内保证只调用一次。以动画为例，人眼中一秒播放超过24张图片就会形成动画，假设有100张图片，我们一秒播放100张过于浪费，一秒播放24张就够了。

防抖:持续触发事件后，时间段内没有再触发事件，才调用一次。以坐电梯为例，电梯10s运行一次。如果快要运行时进来一个人，则重新计时。

```js
//节流
function throttle(fn,delay) {
  let timer=null
  return function () {
    if(!timer){
      timer=setTimeout(()=>{
        fn.call(this,arguments)
        timer=null
      },delay)
    }
  }
}
//防抖
function debounce(fn,delay) {
  let timer=null
  return function () {
    if(timer){
      clearTimeout(timer)
    }
    timer=setTimeout(()=>{
      fn.call(this,arguments)
    },delay)
  }
}
```

### 什么是防抖和节流？有什么区别？如何实现？

**防抖**

>   触发高频事件后n秒内函数只会执行一次，如果n秒内高频事件再次被触发，则重新计算时间

-   思路：

>   每次触发事件时都取消之前的延时调用方法

```js
function debounce(fn) {
      let timeout = null; // 创建一个标记用来存放定时器的返回值
      return function () {
        clearTimeout(timeout); // 每当用户输入的时候把前一个 setTimeout clear 掉
        timeout = setTimeout(() => { // 然后又创建一个新的 setTimeout, 这样就能保证输入字符后的 interval 间隔内如果还有字符输入的话，就不会执行 fn 函数
          fn.apply(this, arguments);
        }, 500);
      };
    }
    function sayHi() {
      console.log('防抖成功');
    }

    var inp = document.getElementById('inp');
    inp.addEventListener('input', debounce(sayHi)); // 防抖
```

**节流**

>   高频事件触发，但在n秒内只会执行一次，所以节流会稀释函数的执行频率

-   思路：

>   每次触发事件时都判断当前是否有等待执行的延时函数

```text
function throttle(fn) {
      let canRun = true; // 通过闭包保存一个标记
      return function () {
        if (!canRun) return; // 在函数开头判断标记是否为true，不为true则return
        canRun = false; // 立即设置为false
        setTimeout(() => { // 将外部传入的函数的执行放在setTimeout中
          fn.apply(this, arguments);
          // 最后在setTimeout执行完毕后再把标记设置为true(关键)表示可以执行下一次循环了。当定时器没有执行的时候标记永远是false，在开头被return掉
          canRun = true;
        }, 500);
      };
    }
    function sayHi(e) {
      console.log(e.target.innerWidth, e.target.innerHeight);
    }
    window.addEventListener('resize', throttle(sayHi));
```

### js为什么需要放在body(更好的回答其实是浏览器的渲染引擎和js解析引擎的冲突，当然回答js是单线程执行也没问题,如何优化)？

### 操作DOM为什么是昂贵的？

### js如何执行(even Loop/宏任务、微任务，事件队列，promise,async/await)？

### **异步回调（如何解决回调地狱）**

```text
promise、generator、async/await
promise： 1.是一个对象，用来传递异步操作的信息。代表着某个未来才会知道结果的时间，并未这个事件提供统一的api，供进异步处理
	  2.有了这个对象，就可以让异步操作以同步的操作的流程来表达出来，避免层层嵌套的回调地狱
	  3.promise代表一个异步状态，有三个状态pending（进行中），Resolve(以完成），Reject（失败）
	  4.一旦状态改变，就不会在变。任何时候都可以得到结果。从进行中变为以完成或者失败
		promise.all() 里面状态都改变，那就会输出，得到一个数组
		promise.race() 里面只有一个状态变为rejected或者fulfilled即输出
		promis.finally()不管指定不管Promise对象最后状态如何，都会执行的操作（本质上还是then方法的特例）
```

### **浏览器事件循环和node事件循环**

浏览器事件循环:

1.  同步任务在主线程执行，在主线程外还有个任务队列用于存放异步任务
2.  主线程的同步任务执行完毕，异步任务入栈，进入主线程执行
3.  上述的两个步骤循环，形成eventloop事件循环 浏览器的事件循环又跟宏任务和微任务有关，两者都属于异步任务。

>   js异步有一个机制，就是遇到宏任务，先执行宏任务，将宏任务放入任务队列，再执行微任务，将微任务放入任务队列，他俩进入的不是同一个任务队列。往外读取的时候先从微任务里拿这个回调函数，然后再从宏任务的任务队列上拿宏任务的回调函数

宏任务:

-   script
-   定时器 setTimeout setInterval setImmediate

微任务:

-   promise
-   process.nextTick()
-   MutationObserver

node事件循环：

1.  timer阶段
2.  I/O 异常回调阶段
3.  空闲预备阶段
4.  poll阶段
5.  check阶段
6.  关闭事件的回调阶段

### js的作用域？

**全局作用域**

>   浏览器打开一个页面时，浏览器会给JS代码提供一个全局的运行环境，那么这个环境就是全局作用域 一个页面只有一个全局作用域，全局作用域下有一个window对象 window是全局作用域下的最大的一个内置对象（全局作用域下定义的变量和函数都会存储在window下） 如果是全局变量，都会给window新增一个键值对；属性名就是变量名，属性值就是变量所存储的值 如果变量只被var过，那么存储值是undefined 在私有作用域中是可以获取到全局变量的，但是在全局作用域中不能获取私有变量

**私有作用域**

>   函数执行会形成一个新的私有的作用域（执行多次，形成多个私有作用域） 私有作用域在全局作用域中形成，具有包含的关系； 在一个全局作用域中，可以有很多个私有作用域 在私有作用域下定义的变量都是私有变量 形参也是私有变量 函数体中通过function定义的函数也是私有的，在全局作用域不能使用；

**块级作用域**

>   es6中新引入的一种作用域 在js中常见到的if{}、for{}、while{}、try{}、catch{}、switch case{}都是块级作用域 var obj = {} //对象的大括号不是块级作用域 块级作用域中的同一变量不能被重复声明（块级下var和function不能重名，否则会报错） 作用域链

**上级作用域**

>   函数在哪里定义，他的上一级作用域就是哪，和函数在哪个作用域下执行没有关系 作用域链：当获取变量所对应的值时，首先看变量是否是私有变量，如果不是私有变量，要继续向上一级作用域中查找，如果上一级也没有，那么会继续向上一级查找，直到找到全局作用域为止；如果全局作用域也没有，则会报错；这样一级一级向上查找，就会形成作用域链 当前作用域没有的，则会继续向上一级作用域查找 当前函数的上一级作用域跟函数在哪个作用域下执行没有关系，只跟函数在哪定义有关（重点）

### **你怎样看待闭包（closure）？**

简单来说闭包就是在函数里面声明函数，本质上说就是在函数内部和函数外部搭建起一座桥梁，使得子函数可以访问父函数中所有的局部变量，但是反之不可以，这只是闭包的作用之一。

另一个作用，则是保护变量不受外界污染，使其一直存在内存中。

在工作中我们还是少使用闭包的好，因为闭包太消耗内存，不到万不得已的时候尽量不使用。

```js
function say667(){
  // Local variable that ends up within closure
  var num = 666;
  var sayAlert = function () {
    alert(num);
  }
  num ++;
  return sayAlert;
}
var sayAlert = say667(); 
sayAlert()//执行结果应该弹出的 667
```

执行 say667()后,say667()闭包内部变量会存在, 而 闭包内部函数的内部变量不会存在.使得 Javascript 的垃圾回收机制 GC 不会收回 say667()所占用的 资 源，因为 say667()的内部函数的执行需要依赖 say667()中的变量。这是对闭包作用的非常直白的描 述.

### 基础类型以及如何判断类型？

### 事件机制以及如何实现一个事件队列？

### js深浅拷贝？

浅克隆: 只是拷贝了基本类型的数据，而引用类型数据，复制后也是会发生引用，我们把这种拷贝叫做“（浅复制）浅拷贝”，换句话说，浅复制仅仅是指向被复制的内存地址，如果原地址中对象被改变了，那么浅复制出来的对象也会相应改变。

深克隆： 创建一个新对象，属性中引用的其他对象也会被克隆，不再指向原有对象地址。 JSON.parse、JSON.stringify()

浅拷贝:

-   concat()
-   Object.assign()
-   slice()
-   手写

```js
function shallowCopy(obj){
  if(typeof obj==='function'&& obj!==null){
    let cloneObj=Array.isArray(obj)?[]:{}
    for(let prop in obj){
      if(obj.hasOwnProperty(prop)){
        cloneObj[prop]=obj[prop]
      }
    }
    return cloneObj
  }
  else{
    return obj
  }
}
```

深拷贝:

-   JSON.stringfy(JSON.parse())

>   上面的方法不能解决循环引用，也不能显示函数或undefined

-   手写深拷贝

```js
var deepClone=(obj,map=new WeakMap())=>{
  if(map.get(obj)){
    return obj
  }

  let newObj;
  if(typeof obj==='object'&& obj!==null){
    map.set(obj,true)
    newObj=Array.isArray(obj)?[]:{};
    for(let item in obj){
      if(obj.hasOwnProperty(item)){
        newObj[item]=deepClone(obj[item])
    }
  }
    return newObj;
  }
  else {
    return obj;
  }
};
```

### 都有哪些方式创建对象，静态和动态,构造函数创建对象优缺点

1：Object构造函数创建

```js
var Person =new Object();
Person.name = 'Jason';Person.age = 21;
```

2.  使用对象字面量表示法来创建对象

```js
var Person={};   //等同于var Person =new Object();
var Person={
name:"Jason",
age:21
}
```

3.  使用工厂模式创建对象

```js
function createPerson(name,age,job)
{ var o = new Object(); 
o.name = name; 
o.age = age; 
o.job = job; 
o.sayName = function()
{  alert(this.name);  };
 return o;
 }
var person1 = createPerson('Nike',29,'teacher');
var person2 = createPerson('Arvin',20,'student');
```

4.  使用构造函数创建对象

```js
function Person(name,age,job)
{ this.name = name; 
this.age = age; 
this.job = job; 
this.sayName = function(){ alert(this.name); }; 
}
var person1 = new Person('Nike',29,'teacher');
var person2 = new Person('Arvin',20,'student');
```

5.  原型创建对象模式

```js
function Person(){}
Person.prototype.name = 'Nike';
Person.prototype.age = 20;
Person.prototype.jbo = 'teacher';
Person.prototype.sayName = function(){ alert(this.name);};
var person1 = new Person();person1.sayName();
```

6.  组合使用构造函数模式和原型模式

```js
function Person(name,age,job)
{ this.name =name; 
this.age = age;
 this.job = job;}
Person.prototype = { 
constructor:Person,
 sayName: function()
{ alert(this.name); };
}
var person1 = new Person('Nike',20,'teacher');
```



4.  

### **js继承方式有哪些？**

>   原型链继承 核心： 将父类的实例作为子类的原型
>
>   构造继承 核心：使用父类的构造函数来增强子类实例，等于是复制父类的实例属性给子类
>
>   实例继承 核心：为父类实例添加新特性，作为子类实例返回
>
>   拷贝继承
>
>   组合继承 核心：通过调用父类构造，继承父类的属性并保留传参的优点，然后通过将父类实例作为子类原型，实现 函数复用
>
>   寄生组合继承 核心：通过寄生方式，砍掉父类的实例属性，这样，在调用两次父类的构造的时候，就不会初始化两次实 例方法/属性，避免的组合继承的缺点

### 你是如何理解原型和原型链的？

>   把所有的对象共用的属性全部放在堆内存的一个对象（共用属性组成的对象），然后让每一个对象的 __proto__存储这个「共用属性组成的对象」的地址。而这个共用属性就是原型，原型出现的目的就是为了减少不必要的内存消耗。而原型链就是对象通过__proto__向当前实例所属类的原型上查找属性或方法的机制，如果找到Object的原型上还是没有找到想要的属性或者是方法则查找结束，最终会返回undefined

**原型**

>   所有的函数数据类型都天生自带一个prototype属性，该属性的属性值是一个对象 prototype的属性值中天生自带一个constructor属性，其constructor属性值指向当前原型所属的类 所有的对象数据类型，都天生自带一个_proto_属性，该属性的属性值指向当前实例所属类的原型

### 原型链、说一下继承 

![image-20201220190454866](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201220190500.png)

继承与原型链息息相关。

`JavaScript` 中没有类的概念的，主要通过原型链来实现继承。通常情况下，继承意味着复制操作，然而 `JavaScript` 默认并不会复制对象的属性，相反，`JavaScript` 只是在两个对象之间创建一个关联（原型对象指针），这样，一个对象就可以通过委托访问另一个对象的属性和函数，所以与其叫继承，委托的说法反而更准确些。

```js
//原型链继承
function Parent(){
    this.property=true;
}
Parent.prototype.getValue=function(){
    return this.property;
}
function Son(){
    this.subProperty=false;
}
Son.prototype=new Parent();
let instance=new Son();
```

原型链继承继承了原型的属性和方法，但是有几个缺点:

1.  原型链中包括引用类型的值时，会被所有实例共享
2.  不能实现子类向超类的构造函数中添加属性

由此产生了借用构造函数继承,解决了原型链继承的缺点，它自身又有缺点:不能实现函数复用

```js
//借用构造函数继承
function Parent(){
    this.property=true;
}
function Son(){
    Parent.call(this);
}
复制代码//组合继承
function Parent(){
    this.property=true;
    this.colors=['red','purple','orange']
}
Parent.prototype.getPro=function(){
    return this.property;
}
function Son(property,name){
    Parent.call(this,property);
    this.name=name;
}
Son.prototype=new Parent()
```

组合继承避免了原型链和借用构造函数的缺陷,是最常用的继承之一

```js
//原型继承
var a = {
  friends : ["yuki","sakura"]
};
var b = Object.create(a);
b.friends.push("ruby");
var c = Object.create(a);
c.friends.push("lemon");
alert(a.friends);//"yuki,sakura,ruby,lemon"
```

原型继承缺点跟原型链继承一样，也是引用类型的属性会被所有实例共享

```js
//寄生式继承,可以类比设计模式的工厂模式
function createAnother(obj){
  var clone = object(obj);
  clone.sayHi = function(){
    console.log("hello");
  };
  return clone;
}
```

寄生式继承不能做到函数复用

```js
//寄生组合式继承
function Parent(name){
    this.name=name;
    this.colors=['red','white','gray']
}
Parent.prototype.getName=function(name){
    this.name=name
}
function Son(name,age){
    Parent.call(this,name);//第二次调用Parent()
    this.age=age;
}
Son.prototype=new Parent()//第一次调用Parent()
Son.prototype.constructor=Son;
Son.prototype.getAge=function(){
    return this.age;
}
```

寄生组合式继承避免了在子实例上创建多余的属性，又能保持原型链不变，还能正常使用instanceof()和isPrototypeOf()，是最理想的继承方式。

es6方法的继承:通过extends实现

```js
class Parent(){
    constructor(){}
}
class Son extends Parent(){
    constructor(){
        super()
    }
}
```

### 数组去重

```js
//方法一 使用ES6的Set
function filterArr(arr) {
  return new Set(arr)
}
//方法二:filter+indexOf()判断，如果数字不是第一次出现则被过滤
function filterArr2(arr){
  let newArr=arr.filter((item,index)=>{
    return arr.indexOf(item)===index
  })
  console.log(newArr)
}
//方法三:双重for循环
function filterArr3(arr){
  let isRepeat,newArr=[];
  for(let i=0;i<arr.length;i++){
    isRepeat=false
    for(let j=i+1;j<arr.length;j++){
      if(arr[i]===arr[j]){
        isRepeat=true
        break
      }
    }
    if(!isRepeat){
      newArr.push(arr[i])
    }
  }
  return newArr
}
//方法四:哈希表
function filterArr4(arr){
  let seen={}
  return arr.filter(function (item) {
    return seen.hasOwnProperty(item)?false:(seen[item]=true)
  });
}
//方法五:sort排序，相同的数字会排在相邻n个位置
function filterArr5(arr){
  let lastArr=[]
  const newArr=arr.sort((a,b)=>{
    return a-b
  })
  for(let i=0;i<newArr.length;i++){
    if(newArr[i]!==newArr[i+1]){
      lastArr.push(newArr[i])
    }
  }
  return lastArr
}
```

ES6的set对象 先将原数组排序，在与相邻的进行比较，如果不同则存入新数组

```text
function unique(arr){
    var arr2 = arr.sort();
    var res = [arr2[0]];
    for(var i=1;i<arr2.length;i++){
        if(arr2[i] !== res[res.length-1]){
        res.push(arr2[i]);
    }
}
return res;
}
利用下标查询
 function unique(arr){
    var newArr = [arr[0]];
    for(var i=1;i<arr.length;i++){
        if(newArr.indexOf(arr[i]) == -1){
        newArr.push(arr[i]);
    }
}
return newArr;
}
```

### **call bind apply 的区别？**

>   call() 和apply()的第一个参数相同，就是指定的对象。这个对象就是该函数的执行上下文。
>
>   call()和apply()的区别就在于，两者之间的参数。
>
>   call()在第一个参数之后的 后续所有参数就是传入该函数的值。
>
>   apply() 只有两个参数，第一个是对象，第二个是数组，这个数组就是该函数的参数。 bind() 方法和前两者不同在于： bind() 方法会返回执行上下文被改变的函数而不会立即执行，而前两者是 直接执行该函数。他的参数和call()相同。



### **箭头函数和普通函数区别**

-   箭头函数没有prototype，所以箭头函数本身没有this
-   箭头函数的this指向在定义的时候继承自外层第一个普通函数的this
-   箭头函数没有arguments，普通函数有
-   使用new调用箭头函数会报错
-   不可以使用yield命令，因此箭头函数不能用作 Generator 函数。

### **Promise处理异步**

>   他是ES6中新增加的一个类（new Promise）,目的是为了管理JS中的异步编程的，所以把他称为“Promise设计模式” new Promise 经历三个状态：padding(准备状态：初始化成功、开始执行异步的任务)、fullfilled(成功状态)、rejected(失败状态)== Promise本身是同步编程的，他可以管理异步操作的（重点），new Promise的时候，会把传递的函数立即执行 Promise函数天生有两个参数，resolve(当异步操作执行成功，执行resolve方法),rejected(当异步操作失败，执行reject方法) then()方法中有两个函数，第一个传递的函数是resolve,第二个传递的函数是reject ajax中false代表同步，true代表异步，如果使用异步，不等ajax彻底完成

### 手写promise 

```js
function myPromise(executor) {
    let self=this;
    self.status='pending';
    self.value=undefined;
    self.reason=undefined;

    function resolve(value) {
        if(self.status==='pending'){
            self.value=value
            self.status="resolved"
        }
    }
    function reject(reason) {
        if(self.status==='pending'){
            self.reason=reason
            self.status=status
        }
    }
    try{
        executor(resolve,reject)
    }
    catch (e) {
        reject(e)
    }
}
```

**手写bind**

```js
Function.prototype.myBind=function(context,...args){
   const fn=this
    args=args?args:[]
    return function newFn(...newFnArgs) {
        if(this instanceof newFn){
            return new fn(...args,...newFnArgs)
        }
        return  fn.apply(context,[...args,...newFnArgs])
    }
}
```

###  **手写call、apply**

```js
Function.prototype.myCall=function(context,...args){
    context=context||window
    args=args?args:[]
    const key=Symbol()
    context[key]=this
    const result=context[key](...args)//通过隐式绑定的方式调用函数
    delete context[key]//删除添加的属性
    return result//返回函数调用的返回值
}

Function.prototype.myApply=function(context,args){
    context=context||window
    args=args||[]
    const key=Symbol()
    context[key]=this
    const result=context[key](...args)
    delete context[key]
    return result
}
```

### 对异步的理解 

js是单线程的，但不能一直等着一张张图片加载完成吧，写一个回调的函数，谁先完成任务就执行谁。

### node的理解以及应用方面，优缺点？

js开发环境  **在线聊天**    **对象数据库顶层的 API**   **队列输入**   **数据流**    **代理**    **股票交易商的数据界面**   **应用监控仪表板**     **系统监控仪表板**    **服务器端 Web 应用**    

*（优点）因为 Node 是基于事件驱动和无阻塞的，所以非常适合处理并发请求，

因此构建在 Node 上的代理服务器相比其他技术实现（如 Ruby）的服务器表现要好得多。

此外，与 Node 代理服务器交互的客户端代码是由 javascript 语言编写的，

因此客户端和服务器端都用同一种语言编写，这是非常美妙的事情。

*（缺点）Node是一个相对新的开源项目，所以不太稳定，它总是一直在变，

而且缺少足够多的第三方库支持。看起来，就像是 Ruby/Rails 当年的样子。

### 伪数组 

定义：

1、拥有length属性，其它属性（索引）为非负整数(对象中的索引会被当做字符串来处理，这里你可以当做是个非负整数串来理解)
2、不具有数组所具有的方法

### 伪数组怎么转化为数组？ 

1.  创建一个新数组，遍历这个伪数组，并将其每一项添加到新数组中。
2.  使用`[].slice.call(obj)`， 数组的`slice()`方法,它返回的是数组，使用`call`或者`apply`指向伪数组
3.  使用扩展运算符`...`，比如使用`[...obj]`，需要保证obj是可迭代
4.  使用`ES6`中数组的新方法 `Array.from`，此种方法，对数据源没有特殊的要求，数据源可以不能迭代

### 怎么判断一个元素是不是数组？ 

1.  instanceof  a instanceof Array; //true

2.  constructor 

    let a = [1,3,4];
    a.constructor === Array;//true

3.  Object.prototype.toString.call()

    let a = [1,2,3]
    Object.prototype.toString.call(a) === '[object Array]';//true

4.  Array.isArray()

    let a = [1,2,3]
    Array.isArray(a);//true

### 除了你说的Array.isArray和Array.from还有哪些数组的方法？ 

var a = Array(3);创建一个数组

ES6 Array.of() 返回由所有参数值组成的数组

ES6 Arrary.from() 将两类对象转为真正的数组

改变原数组的方法(9个):

splice() 添加/删除数组元素

sort() 数组排序

pop() 删除一个数组中的最后的一个元素

shift() 删除数组的第一个元素

push() 向数组的末尾添加元素

unshift()

reverse() 颠倒数组中元素的顺序

ES6: copyWithin() 指定位置的成员复制到其他位置

ES6: fill() 填充数组

不改变原数组的方法(8个):

slice() 浅拷贝数组的元素

join() 数组转字符串

toLocaleString() 数组转字符串

toString() 数组转字符串 不推荐

cancat

ES6扩展运算符`...`合并数组

indexOf() 查找数组是否存在某个元素，返回下标

lastIndexOf() 查找指定元素在数组中的最后一个位置

ES7 includes() 查找数组是否包含某个元素 返回布尔

遍历方法(12个):

ES5：    forEach、every 、some、 filter、map、reduce、reduceRight、   

ES6：    find、findIndex、keys、values、entries

### **splice和slice、map和forEach、 filter()、reduce()的区别**

```text
 1.slice(start,end):方法可以从已有数组中返回选定的元素，返回一个新数组，
 包含从start到end（不包含该元素）的数组方法
	注意：该方法不会更新原数组，而是返回一个子数组
 2.splice():该方法想或者从数组中添加或删除项目，返回被删除的项目。（该方法会改变原数组）
	splice(index, howmany,item1,...itemx)
		·index参数：必须，整数规定添加或删除的位置，使用负数，从数组尾部规定位置
		·howmany参数：必须，要删除的数量，
		·item1..itemx:可选，向数组添加新项目
3.map()：会返回一个全新的数组。使用于改变数据值的时候。会分配内存存储空间数组并返回，forEach（）不会返回数据
4.forEach(): 不会返回任何有价值的东西，并且不打算改变数据，单纯的只是想用数据做一些事情，他允许callback更改原始数组的元素
5.reduce(): 方法接收一个函数作为累加器，数组中的每一个值（从左到右）开始缩减，最终计算一个值，不会改变原数组的值
6.filter(): 方法创建一个新数组，新数组中的元素是通过检查指定数组中符合条件的所有元素。它里面通过function去做处理	
```

### map和forEach的区别

**相同点**

>   都是循环遍历数组中的每一项 forEach和map方法里每次执行匿名函数都支持3个参数，参数分别是item（当前每一项）、index（索引值）、arr（原数组），需要用哪个的时候就写哪个 匿名函数中的this都是指向window 只能遍历数组

**不同点**

>   map方法返回一个新的数组，数组中的元素为原始数组调用函数处理后的值。(原数组进行处理之后对应的一个新的数组。) map()方法不会改变原始数组 map()方法不会对空数组进行检测 forEach()方法用于调用数组的每个元素，将元素传给回调函数.(没有return，返回值是undefined）
>   **注意**：forEach对于空数组是不会调用回调函数的。

### Ajax同步和异步的区别?

1. 通过异步模式，提升了用户体验

2. 优化了浏览器和服务器之间的传输，减少不必要 的数据往返，减少了带宽占用

3. Ajax 在客户端运行，承担了一部分本来由服务器承担的工作，减少了大用户量下的服务器负载。

### Ajax的最大的特点是什么？

Ajax 可以实现动态不刷新（局部刷新） readyState 属性 状态 有 5 个可取值：0=未初始化 ，1=正在加载 2=以加载，3=交互中，4=完成

### Åjax 的缺点？

1、ajax 不支持浏览器 back 按钮。

2、安全问题 AJAX 暴露了与服务器交互的细节。

3、对搜索引擎的支持比较弱。

4、破坏了程序的异常机制。

5、不容易调试。

### **Ajax的四个步骤**

>   1.创建ajax实例
>
>   2.执行open 确定要访问的链接 以及同步异步
>
>   3.监听请求状态
>
>   4.发送请求

### **ajax中get和post请求的区别**

>   get 一般用于获取数据
>   get请求如果需要传递参数，那么会默认将参数拼接到url的后面；然后发送给服务器；
>   get请求传递参数大小是有限制的；是浏览器的地址栏有大小限制；
>   get安全性较低
>   get 一般会走缓存，为了防止走缓存，给url后面每次拼的参数不同；放在?后面，一般用个时间戳
>   post 一般用于发送数据
>   post传递参数，需要把参数放进请求体中，发送给服务器；
>   post请求参数放进了请求体中，对大小没有要求；
>   post安全性比较高；
>   post请求不会走缓存；

### **get/post的区别**

```text
1.get数据是存放在url之后，以？分割url和传输数据，参数之间以&相连； post方法是把提交的数据放在http包的Body中
2.get提交的数据大小有限制，（因为浏览器对url的长度有限制），post的方法提交的数据没有限制
3.get需要request.queryString来获取变量的值，而post方式通过request.from来获取变量的值
4.get的方法提交数据，会带来安全问题，比如登录一个页面，通过get的方式提交数据，用户名和密码就会出现在url上
```

### get请求传参长度的误区、get和post请求在缓存方面的区别

**误区：我们经常说get请求参数的大小存在限制，而post请求的参数大小是无限制的。**

实际上HTTP 协议从未规定 GET/POST 的请求长度限制是多少。对get请求参数的限制是来源与浏览器或web服务器，浏览器或web服务器限制了url的长度。为了明确这个概念，我们必须再次强调下面几点:

-   HTTP 协议 未规定 GET 和POST的长度限制
-   GET的最大长度显示是因为 浏览器和 web服务器限制了 URI的长度
-   不同的浏览器和WEB服务器，限制的最大长度不一样
-   要支持IE，则最大长度为2083byte，若只支持Chrome，则最大长度 8182byte

补充补充一个get和post在缓存方面的区别：

-   get请求类似于查找的过程，用户获取数据，可以不用每次都与数据库连接，所以可以使用缓存。
-   post不同，post做的一般是修改和删除的工作，所以必须与数据库交互，所以不能使用缓存。因此get请求适合于请求缓存。

