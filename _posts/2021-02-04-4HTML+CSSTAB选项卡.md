---
layout:     post           # 使用的布局（不需要改）
title:      4HTML+CSSTAB选项卡
subtitle:   4HTML+CSSTAB选项卡 #副标题
date:       2021-02-04             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端

---

## [4HTML+CSSTAB选项卡](https://www.bilibili.com/video/BV1uK411A7S6)

## 效果

![4TAB选项卡](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210204185801.gif)





## HTML

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TAB选项卡</title>
  <link rel="stylesheet" href="css/index.css">
</head>
<body>
  <!-- .tab>input:radio[name="tab" id="tab$"]*5 -->
  <div class="tab">
    <input type="radio" name="tab" id="tab1" checked>
    <input type="radio" name="tab" id="tab2">
    <input type="radio" name="tab" id="tab3">
    <input type="radio" name="tab" id="tab4">
    <input type="radio" name="tab" id="tab5">
    <!-- label[for="tab$"]*5>img[src="images/$.png"] -->
    <label for="tab1"><img src="images/1.png" alt="">HTML</label>
    <label for="tab2"><img src="images/2.png" alt="">CSS</label>
    <label for="tab3"><img src="images/3.png" alt="">JavaScript</label>
    <label for="tab4"><img src="images/4.png" alt="">Vue</label>
    <label for="tab5"><img src="images/5.png" alt="">React</label>
    <!-- ul>li*5>img[src="images/$.png"]+h2+p -->
    <ul>
      <li>
        <img src="images/1.png" alt="">
        <h2>HTML</h2>
        <p>HTML 称为超文本标记语言，是一种标识性的语言。它包括一系列标签．通过这些标签可以将网络上的文档格式统一，使分散的 Internet 资源连接为一个逻辑整体。HTML 文本是由 HTML 命令组成的描述性文本，HTML 命令可以说明文字，图形、动画、声音、表格、链接等。</p>
      </li>
      <li>
        <img src="images/2.png" alt="">
        <h2>CSS</h2>
        <p>层叠样式表(英文全称：Cascading Style Sheets)是一种用来表现 HTML（标准通用标记语言的一个应用）或 XML（标准通用标记语言的一个子集）等文件样式的计算机语言。CSS 不仅可以静态地修饰网页，还可以配合各种脚本语言动态地对网页各元素进行格式化。</p>
      </li>
      <li>
        <img src="images/3.png" alt="">
        <h2>JavaScript</h2>
        <p>JavaScript（简称“JS”）是一种具有函数优先的轻量级，解释型或即时编译型的高级编程语言。虽然它是作为开发 Web 页面的脚本语言而出名的，但是它也被用到了很多非浏览器环境中，JavaScript 基于原型编程、多范式的动态脚本语言，并且支持面向对象、命令式和声明式（如函数式编程）风格。</p>
      </li>
      <li>
        <img src="images/4.png" alt="">
        <h2>Vue</h2>
        <p>VUE 是 iOS 和 Android 平台上的一款 Vlog 社区与编辑工具，允许用户通过简单的操作实现 Vlog 的拍摄、剪辑、细调、和发布，记录与分享生活。还可以在社区直接浏览他人发布的 Vlog，与 Vloggers 互动。</p>
      </li>
      <li>
        <img src="images/5.png" alt="">
        <h2>React</h2>
        <p>React 起源于 Facebook 的内部项目，因为该公司对市场上所有 JavaScript MVC 框架，都不满意，就决定自己写一套，用来架设 Instagram 的网站。做出来以后，发现这套东西很好用，就在 2013 年 5 月开源了。</p>
      </li>
    </ul>
  </div>
</body>
</html>
```

## CSS

```css
* {
  padding: 0;
  margin: 0;
  box-sizing: border-box;
}
body {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #282c34;
}
.tab {
  width: 700px;
  height: 250px;
  color: #607291;
  background-color: #fff;
  overflow: hidden;
}
input {
  display: none;
}
label {
  float: left;
  width: 140px;
  height: 40px;
  line-height: 40px;
  text-align: center;
  font-size: 14px;
  font-weight: 700;
  background-color: #e5e9ea;
  transition: all .3s;
}
label:hover {
  background-color: #fff;
}
label img {
  width: 20px;
  height: 20px;
  vertical-align: middle;
  margin-top: -5px;
  margin-right: 5px;
}
ul{
  /* 清除浮动 */
  clear: both;
  width: 3500px;
  height: 210px;
  transition: all .5s;
}
ul li{
  float: left;
  list-style: none;
  width: 700px;
  height: 210px;
  padding: 40px;
}
ul li img{
  float: left;
  width: 130px;
  height: 130px;
  margin-right: 20px;
}
ul li p{
  /* 首行缩进2个字符 */
  text-indent: 2em;
  /* 伸缩盒子模型 */
  display: -webkit-box;
  /* 在伸缩盒子模型里让子元素垂直排列 */
  -webkit-box-orient: vertical;
  /* 这个不是css的标准语句，需要配合以上两种属性一起使用，意思是只显示三行 */
  -webkit-line-clamp: 3;
  /* 溢出隐藏 */
  overflow: hidden;
  /* 隐藏的文字呈现省略号 */
  margin-top: 20px;
}
#tab1:checked~ul{
  margin-left: 0;
}
#tab2:checked~ul{
  margin-left: -700px;
}
#tab3:checked~ul{
  margin-left: -1400px;
}
#tab4:checked~ul{
  margin-left: -2100px;
}
#tab5:checked~ul{
  margin-left: -2800px;
}
#tab1:checked~label[for="tab1"]{
  background-color: #fff;
}
#tab2:checked~label[for="tab2"]{
  background-color: #fff;
}
#tab3:checked~label[for="tab3"]{
  background-color: #fff;
}
#tab4:checked~label[for="tab4"]{
  background-color: #fff;
}
#tab5:checked~label[for="tab5"]{
  background-color: #fff;
}
```

## note

