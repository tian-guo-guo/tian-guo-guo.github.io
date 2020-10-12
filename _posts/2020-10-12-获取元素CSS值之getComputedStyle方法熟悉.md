---
layout:     post           # 使用的布局（不需要改）
title:      获取元素CSS值之getComputedStyle方法熟悉           # 标题 
subtitle:   获取元素CSS值之getComputedStyle方法熟悉           #副标题
date:       2020-10-12             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [获取元素CSS值之getComputedStyle方法熟悉](https://www.jianshu.com/p/7fa5d66a5c4b)

### 一、碎碎念~前言

我们都用过jQuery的`CSS()`方法，其底层运作就应用了getComputedStyle以及getPropertyValue方法。

对于那些只想混口饭吃的人来讲，晓得`CSS()`如何使用就足够了。对于希望在JS道路上越走越远的来人说，简单了解一些JS库底层实现对自己的学习很有帮助。可能谈不上信手拈来的使用，至少对创造一些创意新颖的新技术拓宽了思路。

jQuery为何受欢迎，其中原因之一就是方法名称比较短。好比打架一样，块头大的潜意识认为厉害，就不由得心生畏惧，退避三舍；小个子（或村妇，小孩）嘛，自然以为是软豆腐，愿意接受作为对手。

因此，当看到类似`getComputedStyle`这么长的方法名的时候，我们身体里那个懒惰小人突然就醒了：哎哟哟，这东西，就跟放心食品一样，几年才见一回。看上去又蛮复杂，想到晚上还要跟妹子聊天。这东西，就让它从旁边过吧，反正不会掉块肉的。

网上不是有这么个段子嘛：**可怕的是，比你聪明的人比你更勤奋**。那么，这里的“勤奋”的差异体现在什么地方呢？就拿这个`getComputedStyle`举例：懒惰青年是看一下，立马像看见凤姐一样视线离开；普通青年是看一下，发现自己不了解，百一下或谷一下，熟悉了解之；勤奋青年是不仅了解，还抽出时间实践之（如做个简单demo测试），熟悉方法的特异性，验证一些观点正确性。

您可以按照你现在的心理状态看看你是哪类青年：如果此时，您已经对本文的内容没有兴趣了（当然，您的工作与JS关系亲密），您是……（你懂的）；如果您看完本文内容，发现，还是有不少收获，心中回味下，然后ctrl+w去其他地方觅食，不再回来，那您是普通青年；如果您看完本文，然后对所说的一些内容作了额外的测试，例如IE9浏览器是否支持测试等，并通过评论形式指出可能的错误，不得不说您是勤奋青年，假以时日，必有一番技术作为。从我文章的些评论来看，有不少同行就是这样的勤奋人儿。

の, 好久没写文章，话又多了，打住，进入正题。

### 二、getComputedStyle是？

`getComputedStyle`是一个可以获取当前元素所有最终使用的CSS属性值。返回的是一个CSS样式声明对象([object CSSStyleDeclaration])，只读。

>   `getComputedStyle()` gives the final used values of all the CSS properties of an element.

**语法**如下：

```
var style = window.getComputedStyle("元素", "伪类");
```

例如：

```
var dom = document.getElementById("test"),
    style = window.getComputedStyle(dom , ":after");
```

就两个参数，大家都懂中文的，没什么好说的。只是额外提示下：Gecko 2.0 (Firefox 4 / Thunderbird 3.3 / SeaMonkey 2.1) 之前，第二个参数“伪类”是必需的（如果不是伪类，设置为`null`），不过现在嘛，不是必需参数了。

### 三、getComputedStyle与style的区别

我们使用`element.style`也可以获取元素的CSS样式声明对象，但是其与`getComputedStyle`方法还有有一些差异的。

1.  **只读与可写**
    正如上面提到的`getComputedStyle`方法是只读的，只能获取样式，不能设置；而`element.style`能读能写，能屈能伸。
2.  **获取的对象范围**
    `getComputedStyle`方法获取的是最终应用在元素上的所有CSS属性对象（即使没有CSS代码，也会把默认的祖宗八代都显示出来）；而`element.style`只能获取元素`style`属性中的CSS样式。因此对于一个光秃秃的元素`<p>`，`getComputedStyle`方法返回对象中`length`属性值（如果有）就是`190+`(据我测试FF:192, IE9:195, Chrome:253, 不同环境结果可能有差异), 而`element.style`就是`0`。

### 四、getComputedStyle与defaultView

如果我们查看jQuery源代码，会发现，其`css()`方法实现不是使用的`window.getComputedStyle`而是`document.defaultView.getComputedStyle`，唷？这是怎么一回事？
![jQuery源码使用document.defaultView.getComputedStyle截图证明](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_211408.png)

实际上，使用`defaultView`基本上是没有必要的，`getComputedStyle`本身就存在`window`对象之中。根据[DennisHall](https://developer.mozilla.org/User:DennisHall)的说法，使用`defaultView`可能一是人们不太乐意在window上专门写个东西，二是让API在Java中也可用（这我不懂，忘指点~~）。

不过有个特殊情况，在FireFox3.6上不使用`defaultView`方法就搞不定的，就是[访问框架(frame)的样式](https://github.com/jquery/jquery/pull/524).

### 五、getComputedStyle兼容性

对于桌面设备：

|              |                         Chrome                         |                    Firefox (Gecko)                     |                      Internet Explorer                      |                            Opera                            |                         Safari                         |
| :----------: | :----------------------------------------------------: | :----------------------------------------------------: | :---------------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------------------: |
|   基本支持   | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |                              9                              |   ![支持](https://www.zhangxinxu.com/study/image/ok.gif)    | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |
| 伪类元素支持 | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![不支持](https://www.zhangxinxu.com/study/image/error.gif) | ![不支持](https://www.zhangxinxu.com/study/image/error.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |

对于手机设备：

|            |                        Android                         |                 Firefox Mobile (Gecko)                 |                          IE Mobile                          |                      Opera Mobile                      |                     Safari Mobile                      |
| :--------: | :----------------------------------------------------: | :----------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: |
|  基本支持  | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |                          WP7 Mango                          | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |
| 伪元素支持 |                           ?                            |                           ?                            | ![不支持](https://www.zhangxinxu.com/study/image/error.gif) |                           ?                            |                           ?                            |

上面打问号的表示没有测试，是否兼容不知。如果您方便测试，欢迎将测试结果告知，这里将及时更新，并附上您的姓名，以谢您做的贡献。

我们先把注意力放在桌面设备上，可以看到，`getComputedStyle`方法IE6~8是不支持的，得，背了半天的媳妇，发现是孙悟空变的——郁闷了。不急，IE自有自己的一套东西。

### 六、getComputedStyle与currentStyle

`currentStyle`是IE浏览器自娱自乐的一个属性，其与`element.style`可以说是近亲，至少在使用形式上类似，`element.currentStyle`，差别在于`element.currentStyle`返回的是元素当前应用的最终CSS属性值（包括外链CSS文件，页面中嵌入的`<style>`属性等）。

因此，从作用上将，`getComputedStyle`方法与`currentStyle`属性走的很近，形式上则`style`与`currentStyle`走的近。不过，`currentStyle`属性貌似不支持伪类样式获取，这是与`getComputedStyle`方法的差异，也是jQuery `css()`方法无法体现的一点。

//zxx: 如果你只知jQuery css()方法，你是不会知道伪类样式也是可以获取的，虽然部分浏览器不支持。

例如，我们要获取一个元素的高度，可以类似下面的代码：

```
alert((element.currentStyle? element.currentStyle : window.getComputedStyle(element, null)).height);
```

您可以狠狠地点击这里：[使用getComputedStyle和currentStyle 获取元素高度demo](http://www.zhangxinxu.com/study/201205/currentstyle-getcomputedstyle-element-height.html)

结果FireFox下显示`24px`(经过计算了), 而IE浏览器下则是CSS中的`2em`属性值：
![Firefox下显示的计算后的24px值 张鑫旭-鑫空间-鑫生活](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_220826.png) ![IE9下显示的CSS中的2em值 张鑫旭-鑫空间-鑫生活](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_220905.png)

`getComputedStyle`方法与`currentStyle`属性其他具体差异还有很多，我以一个普通按钮做元素，遍历了其中靠谱的属性名和属性值，您可以狠狠地点击这里：[getComputedStyle和currentStyle属性展示demo](http://www.zhangxinxu.com/study/201205/currentstyle-getcomputedstyle-test.html)

仔细对比查看，我们可以看到不少差异，例如浮动属性，FireFox浏览器下是这个(`cssFloat`)：
![FireFox下的浮动属性名 ](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_221607.png)

IE7浏览器下则是`styleFloat` ：
![IE7浏览器下的styleFloat属性 张鑫旭-鑫空间-鑫生活](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_221737.png)

而IE9浏览器下则是`cssFloat`和`styleFloat`都有。

等其他N多差异。

### 七、getPropertyValue方法

`getPropertyValue`方法可以获取CSS样式申明对象上的属性值（直接属性名称），例如：

```
window.getComputedStyle(element, null).getPropertyValue("float");
```

如果我们不使用`getPropertyValue`方法，直接使用键值访问，其实也是可以的。但是，比如这里的的`float`，如果使用键值访问，则不能直接使用`getComputedStyle(element, null).float`，而应该是`cssFloat`与`styleFloat`，自然需要浏览器判断了，比较折腾！

使用`getPropertyValue`方法不必可以驼峰书写形式（不支持驼峰写法），例如：`style.getPropertyValue("border-top-left-radius")`;

**更新于2018-05-30**

`getPropertyValue`方法可以用来获取CSS原生变量值，如下示意：

```
const sidebarElement = document.querySelector('.sidebar');

// cssStyles包含sidebarElement元素的样式
const cssStyles = getComputedStyle(sidebarElement);

// 获取 --left-pos CSS 变量值
const cssVal = String(cssStyles.getPropertyValue('--left-pos')).trim(); 

// 输出cssVal
// 输出变量值是：100px 
console.log(cssVal);
```

设置CSS变量可以使用`setProperty()`方法，例如：

```
sidebarElement.style.setProperty('--left-pos', '200px');
```

**兼容性**
`getPropertyValue`方法IE9+以及其他现代浏览器都支持，见下表：

|          |                         Chrome                         |                    Firefox (Gecko)                     | Internet Explorer |                         Opera                          |                         Safari                         |
| :------: | :----------------------------------------------------: | :----------------------------------------------------: | :---------------: | :----------------------------------------------------: | :----------------------------------------------------: |
| 基本支持 | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |         9         | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) | ![支持](https://www.zhangxinxu.com/study/image/ok.gif) |

OK，一涉及到兼容性问题(IE6-8肿么办)，感觉头开始微微作痛了~~，不急，IE自由一套自己的套路，就是`getAttribute`方法。

### 八、getPropertyValue和getAttribute

在老的IE浏览器（包括最新的），`getAttribute`方法提供了与`getPropertyValue`方法类似的功能，可以访问CSS样式对象的属性。用法与`getPropertyValue`类似：

```
style.getAttribute("float");
```

注意到没，使用`getAttribute`方法也不需要`cssFloat`与`styleFloat`的怪异写法与兼容性处理。不过，还是有一点差异的，就是属性名需要驼峰写法，如下：

```
style.getAttribute("backgroundColor");
```

如果不考虑IE6浏览器，貌似也是可以这么写：

```
style.getAttribute("background-color");
```

实例才是王道，您可以狠狠地点击这里：[getPropertyValue和getAttribute获取背景色demo](http://www.zhangxinxu.com/study/201205/getpropertyvalue-getAttribute-background-color.html)

结果FireFox下一如既往的rgb颜色返回(Chrome也是返回rgb颜色)：
![FireFox浏览器下一如既往的RGB颜色返回 张鑫旭-鑫空间-鑫生活](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_230119.png)

对于IE9浏览器，虽然应用的是`currentStyle`, 但是从结果上来讲，`currentStyle`返回的对象是完全支持`getPropertyValue`方法的。
![IE9下截图](http://image.zhangxinxu.com/image/blog/201205/2012-05-08_230242.png)

### 九、getPropertyValue和getPropertyCSSValue

从长相上看`getPropertyCSSValue`与`getPropertyValue`是近亲，但实际上，`getPropertyCSSValue`要顽劣的多。

`getPropertyCSSValue`方法返回一个CSS最初值(CSSPrimitiveValue)对象(width, height, left, …)或CSS值列表(CSSValueList)对象(backgroundColor, fontSize, …)，这取决于`style`属性值的类型。在某些特别的style属性下，其返回的是自定义对象。该自定义对象继承于CSSValue对象（就是上面所说的`getComputedStyle`以及`currentStyle`返回对象）。

`getPropertyCSSValue`方法兼容性不好，IE9浏览器不支持，Opera浏览器也不支持（实际支持，只是老是抛出异常）。而且，虽然FireFox中，`style`对象支持`getPropertyCSSValue`方法，但总是返回`null`. 因此，目前来讲，`getPropertyCSSValue`方法可以先不闻不问。

### 十、补充~结语

有了jQuery等优秀库，我们有熟悉底层的`getComputedStyle`方法的必要吗？

实际上，本文一直没有深入展开`getComputedStyle`方法一个很重要的，类似`css()`方法没有的功能——获取伪类元素样式。但从这一点上将，熟悉`getComputedStyle`方法有必要。

下一篇文章，我就将介绍如何实现`getComputedStyle`方法在伪类元素上的特异功能，实现CSS3 media queries下一些JS交互，及实际应用。

well, 开篇已经啰哩吧嗦不少内容了，这里就不再讲废话了。行文匆忙，文中难免有表述不准确之处，欢迎指正。欢迎补充，感谢阅读，希望本文的内容能够对您的学习有所帮助。

本文为原创文章，转载请注明来自[张鑫旭-鑫空间-鑫生活](http://www.zhangxinxu.com/)[[http://www.zhangxinxu.com](http://www.zhangxinxu.com/)]
本文地址：http://www.zhangxinxu.com/wordpress/?p=2378