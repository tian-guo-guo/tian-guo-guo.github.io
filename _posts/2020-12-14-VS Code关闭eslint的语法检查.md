---
layout:     post           # 使用的布局（不需要改）
title:      VS Code关闭eslint的语法检查
subtitle:   VS Code关闭eslint的语法检查  #副标题
date:       2020-12-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - vue
 
---

# VS Code关闭eslint的语法检查

## VS Code

在文件->首选项->设置中添加`"eslint.enable": false`配置即可

![img](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201214212656.png)

image.png

右侧用户设置会自动覆盖左侧的默认设置

## vue工程中

在webpack.base.conf.js配置文件中删除有关`loader: 'eslint-loader',`的配置，如下：

```javascript
const createLintingRule = () => ({
  test: /\.(js|vue)$/,
  loader: 'eslint-loader',
  enforce: 'pre',
  include: [resolve('src'), resolve('test')],
  options: {
    formatter: require('eslint-friendly-formatter'),
    emitWarning: !config.dev.showEslintErrorsInOverlay
  }
})
```