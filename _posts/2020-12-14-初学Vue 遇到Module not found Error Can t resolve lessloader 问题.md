---
layout:     post           # 使用的布局（不需要改）
title:      初学Vue 遇到Module not found Error Can t resolve lessloader 问题查
subtitle:   初学Vue 遇到Module not found Error Can t resolve lessloader 问题  #副标题
date:       2020-12-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - vue
 
---

# 初学Vue 遇到Module not found Error Can t resolve lessloader 问题

```python
<span style="font-family:Consolas, Inconsolata, Courier, monospace;">./src/components/1-模板语法/test.vueModule not found: Error: Can't resolve 'less-loader' in 'D:\vue\myVue\myVue\src\components\1-模板语法' @ ./src/components/1-模板语法/test.vue 4:2-315 @ ./node_modules/babel-loader/lib!./node_modules/vue-loader/lib/selector.js?type=script&index=0!./src/components/1-模板语法/HelloWorld.vue @ ./src/components/1-模板语法/HelloWorld.vue @ ./src/router/index.js @ ./src/main.js @ multi (webpack)-dev-server/client?http://localhost:8080 webpack/hot/dev-server ./src/main.js</span>
```

一、学习vue时，导入一个子组件时遇到Module not found:Error:Can`t resolve 'less-loader' 问题，实际上是在子组件中的样式里加了这么个代码

```html
<style lang="less" scoped>
</style>
```

而这个less是需要安装的，npm install --save-dev less-loader less

这是他npm的地址

https://www.npmjs.com/package/less-loader

二、记住！ 但凡只要遇到 Module not found:Error:Can`t resolve 'XXXXXXXXXXXX'，都是项目中没有安装这个依赖。

只需要执行 ： npm install  XXXXXX  就能解决问题。

或许初学时并不知道这个依赖是干嘛的，放心，随着经验不断积累，慢慢就知道些是干嘛的了

这个是[NPM仓库](https://www.npmjs.com/)，所有需要的依赖都会从这里下载，并且提供了安装和使用的方法。

 