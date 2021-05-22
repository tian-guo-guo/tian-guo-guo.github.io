---
layout:     post           # 使用的布局（不需要改）
title:      VS Code 插件之 vscode-debug-visualizer
subtitle:   debug visualizer  #副标题
date:       2021-05-22             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
 
---

## [VS Code 插件之 vscode-debug-visualizer](https://www.cnblogs.com/fws407296762/p/13993640.html)



>   最近在网上看到有人在推 vscode-debug-visualizer 这个神器，想了解一下做个对比

## 介绍

在我们写代码得时候，最麻烦的事是出现错误很难定位到问题，特别是代码特别长的时候，错误隐藏的很深的时候，不管是 `debugger` 还是 `console.log`，亦或用浏览器的调试工具打断点，都是需要慢慢一条一条的排错。

这些调试方式都是需要我们大脑去思考、去排错，那有没有一种方式更方便，能将结果以图的形式展示给我们看，并且将过程展示给我们看呢？

这里我们介绍一个神器：[vscode-debug-visualizer](https://github.com/hediet/vscode-debug-visualizer)

它提供了一种以图表的形式展示数据结构形成程的过程，有树形、表格、曲线、图等。

这种展示形式，不仅可以帮我们在代码调试的时候用到，而且在学习数据结构、算法、刷 Leecode 的时候也可以用到，它会让你很清晰的看到数据生成的过程。

## 使用方式

#### 安装

首先在 VS Code 的插件库中搜索 `debug-visualizer`, 然后安装。

#### 使用步骤

-   打开需要调试的代码文件
-   选择需要调试的地方打上断点
-   启动调试
-   `Ctrl + Shift + P` 打开命令面板，输入 `Debug Visualizer: New View` 打开一个新的可视化窗口
-   在可视化窗口输入需要展示的数据表达式
-   按 `F10` 开始调试，在可视化窗口中就是展示出数据的图表

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154500.gif)

一个可视化窗口只能调试一个文件，可以打开多个窗口同时调试。如果按下 `Shift+ F1` 或者输入命令：`Debug Visualizer: Use Selection as Expression` 就是调试选中的文本

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154512.gif)

#### 输入规则

可能到这里大家对可视化窗口中输入框中输入的内容有点好奇，这里面到底能输入什么。图表展示是必须要用 `JSON` 数据才能展示，大家可以到 [Visualization Playground](https://hediet.github.io/visualization/?darkTheme=1) 这里看看，可以有哪些 `JSON` 类型的数据

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154522.png)
![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154530.png)
![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154538.png)
![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154549.png)
![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210522154600.png)

而输入框里面，我们可以输入 `变量名`、`数组`、`函数`、`JSON` 三种类型的数据，然后插件内部会自动换换成对应的图表 `JSON` 数据，如果实在转换不了就会执行 `toString`

### API

我们有两种方式可以开始调试，一种方式是在输入框中输入调试规则代码，还有一种是在代码内部写调试规则代码。

如果是在输入框中，我们可以使用插件暴露出来的 `hedietDbgVis` 全局对象。

`hedietDbgVis`对外提供了 7 个方法：`createGraph`、`createGraphFromPointers`、`tryEval`、`markedGrid`、`cache`、`asData`、`getApi`。这几个都是插件中的 `helper`。

我们要注意一下，提供出来的几个 `helper` 都是针对复杂的数据类型，例如链表、双向链表、树、图之类的，如果是数据结构相对简单，出现的情况可能不是很理想。

我想重点讲讲 `createGraphFromPointers`，因为其实在 `createGraphFromPointers` 中执行了 `createGraph`，两个函数是一样的，只是在 `createGraphFromPointers` 中做了一些数据处理

#### createGraphFromPointers

这个主要是用来画图表的，提供两个参数，一个是代码中的变量，如果你的代码结构如下

```
class LinkedList {
	constructor() {
		this.head = null;
	}
}

class Node {
	constructor(data, next = null) {
		(this.data = data), (this.next = next);
	}
}

LinkedList.prototype.insertAtBeginning = function(data) {
	let newNode = new Node(data);
	newNode.next = this.head;
	this.head = newNode;
	return this.head;
};

const list = new LinkedList();

list.insertAtBeginning("4");
list.insertAtBeginning("3");
list.insertAtBeginning("2");
list.insertAtBeginning("1");
```

我们的调试代码的写法是：

```
hedietDbgVis.createGraphFromPointers(
	hedietDbgVis.tryEval([
		"list.head",
		"newNode",
		"node",
		"previous",
		this.constructor.name === "LinkedList" ? "this.head" : "err",
	]),
	n => ({
		id: n.data,
		color: "lightblue",
		label: `${n.data}`,
		edges: [{ to: n.next, label: "next" }].filter(i => !!i.to),
	})
)
```

大家可以看到 `createGraphFromPointers` 传入两个参数，第一个参数是需要将代码中的那些变量传递到图表中，第二个参数是获取第一个参数中的数据作为图表的配置。例如上面的意思：

```
id: n.data  获取第一个参数中每一项中的 data
color: "lightblue" 图表颜色
label: `${n.data}` 图表上显示什么内容
edges: [{ to: n.next, label: "next" }].filter(i => !!i.to)
这个就是图表下一个节点显示什么内容，如果加了这个，这个数据中的每一项肯定是对象了
```

#### getDataExtractorApi

还有这个函数，主要是用来自定义数据提取的，因为 `createGraphFromPointers` 已经定死了用的是 `Graph` 类型的图表，如果我们不想用这个图表怎么弄？就可以用到 `getDataExtractorApi`

## 结论

使用了差不多两天的时间，实际上给我的感觉并不是很好：

-   API 文档不详细，很多都没说，要自己研究
-   提供出来展示的形式很少，大部分用的就是折线图和关系树图
-   由于提供的图较少，因此我们可以调试的类型不是很多，大部分就是数组，数组嵌套对象

这个神器在用折线图表现数组的时候确实非常直观，非常推荐大家在做一些矩阵或者做一些三维计算的时候，还有一些动态规划算法的时候，可以用到，比较直观

还有关系树图用来学习链表、双向链表、图、树的算法和数据结构的时候，完全就是神器，以前要自己大脑去画图，现在直接看就行了

也可能是我才疏学浅，了解的不够多，如果大家有新的发现，可以给我留言