---
layout:     post           # 使用的布局（不需要改）
title:      guoguomusic
subtitle:   guoguomusic #副标题
date:       2021-01-20             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试

---

# Vue player

最终效果：http://autumnfish.cn/webmusic/

# ES6

## 01.let

和var关键字的用法基本一致
没有变量提升

## 02.模板字符串

使用、包裹文本
在需要替换的位置使用${}占位，并填入内容即可

## 03.对象简化赋值

如果属性名和变量名相同，可以简写
方法可以省略function关键字

## 04.箭头函数

function省略掉，替换为=>
参数只有一个时，可以省略()
函数体只有一行时，可以省略{ }
函数体只有一行，并且有返回值时，如果省略了{}，必须省略return

## 05.箭头函数中的this

创建时的this是谁，运行的时候this就是谁

# 回顾

*需求：*

​        *1. 回车搜歌*

​            *接口1:歌曲搜索*

​            *地址:https://autumnfish.cn/search*

​            *方法:get*

​            *参数:keywords  搜索关键字*

​        *2. 列表渲染*

​        *3. 点击播放*

​            *接口2:获取歌曲播放地址*

​            *地址:https://autumnfish.cn/song/url*

​            *方法:get*

​            *参数:id  歌曲id*

# 单文件组件

后缀名是. vue，可以同时编写结构，逻辑和样式
template标签中写结构
script标签中写逻辑
style 标签中写样式
安装了vetur
插件可以通过<vue生成基础结构

# 快速原型开发

## 环境配置

安装Node.js全部使用默认的设置，- 路下一步即可
打开命令行工具，输入命令node -V以及npm -V检查是否安装成功
通过命令npm install -g @vue/cli-service-global安装一个小工具

## 基本使用

保证环境配置成功之后
在.vue文件所在的路径下打开终端输入vue serve文件名
等待解析，然后在浏览器访问出现的地址即可
注意:

1. template中必须有一个根节点
2. script中的data写成函数，内部返回一个对象
3. 如果要使用写好的样式，直接import即可

使用npm下载别人写好的包(模块，库)
比如axiQs，下载的命令是npm install  axios或者缩写为npm.i axios
在要使用的组件中，使用import导入下载的包(模块，库)
比如axios:导入的代码是import axios from 'axios'
开心的使用刚刚导入的包(模块，库)吧

## 组件抽取

创建components文件夹创建一个 文件叫做player.vue,
把播放器的代码剪切进去，并调整图片、css 文件的路径 
App.vue中使用import导入播放器组件
在components中添加(注册)播放器组件
页面上使用注册的名字当做标签名即可使用播放器组件

通过chrome的vue插件可以更好地检查，以及调试代码
直接拖入到chrome中的更多工具扩展程序重启浏览器即可
在使用vue的项目中插件会自动亮起，在开发者界面中通过vue分栏即可使用

# vue-cli

帮你创建项目基本结构的工具
帮你整合了很多东西，
用得到的，也许用得到的全部整合了
我们不在需要自己-步-步的搭建这些了

## 01.环境配置

保证Node.js安装成功的情况下
通过命令npm install -g @vue/cli安装一个小工具
如果失败了:
先输入nRm install -g cnpm安装一个小工 具(cnpm)
成功之后再输入cnpm install -g @vue/cli通过刚刚安装的工具来安装vue-cli

## 02.项目创建及运行

在想要创建项目的文件夹下输入vue create项目名
调整一下设置,
然后回车
如果成功了，依次输入最后出现的提示代码
稍等片刻，
在浏览器中输入出现的地址即可访问

## 03.文件结构

自己创建，或者用创建好的项目，然后运行起来
重点关注src文件夹
public目录下可以替换图标

## 04.头部组件

重点学习vue-cli创建的项目咋写代码
今天用图片，明天在用真实的标签
输入框有逻辑，用标签替代

## 05.主体组件

重点学习vue-cli创建的项目咋写代码
导航有逻辑，用标签来实现
中间部分需要点击切换，先个盒子占位

## 06.发现音乐

重点学习vue-cli创建的项目咋写代码
因为这里要切换，并且也有自己的逻辑，弄个组件丢进去，截图走起
其他的页面也需要切换哦，路由走起 

# vue-router

管理多个组件切换关系的大杀器
用它可以做SPA
Single Page Application单页面应用

## 01.安装

在项目的根目录打开终端
通过命令npm install vue-router下载
main.js中
1.导入
2.use- -下
3.创建路由
4.挂载到Vue实例上

## 02.配置规则

通过routes属性配置地址和路由管理的组件关系
main.js中
1.导入组件
2.routes属性中进行配置关系
3.path:设置地址
4.component:设置组件
5.可以配置多个

## 03.路由出口

希望匹配到的组件显示在哪里，就在哪里设置一个router-view标签
多配置几组对应关系.
discovery:发现音乐
playlists:推荐歌单
songs:最新音乐
mvs:最新mv

## 04.声明式导航

通过router-link标签设置to属性为地址可以实现点击切换

## 05.编程式导航

组件中通过this.$router.push(地址)可以通过代码实现切换

## 06.路由传参

在地址的后面写上?分隔
通过key=value&key2=value的方式添加参数
组件中通过this.$route.query访问对应的key即可获取数据

# Element-ui

 饿了么前端推出的Vue组件库
很多常用的功能已经写好了，直接用

## 01.安装

在项目的根目录打开终端.
通过命令
npm i element-ui下载
https://element.eleme.cn/#/zh-CN/component/installation
main.js中
1.导入Element-ui
2.导入样式
3.use--下
https://element.eleme.cn/#/zh-CN/component/quickstart

## 02.按钮

看文档
https://element.eleme.cn/#/zh-CN/component/button

## 03.消息提示

看文档
https://element.eleme.cn/#/zh-CN/component/message

## 04.tab切换

看文档
https://element.eleme.cn/#/zh-CN/component/tabs

## 05.分页

看文档
https://element.eleme.cn/#/zh-CN/componentpagination

## 06.轮播图

看文档
https://element.eleme.cn/#/zh-CN/component/carousel



# 播放器

## 发现音乐

项目已经创建完毕
布局已经写好，
路由和Element-ui已经整合完毕
项目跑起来(npm run serve)
基础模板在02-其他资料/播放器布局中

### 01.轮播图

安装并导入axios
在created生命周期函数中调用轮播图接口
获取到数据并渲染到页面上

### 02.推荐歌单

axios已经下载了，直接使用即可
在created生命周期函数中调用推荐歌单接口
获取到数据并渲染到页面上

### 03.最新音乐

在created生命周期函数中调用最新音乐接口
获取到数据并渲染到页面上

### 04.点击播放

为播放按钮绑定点击事件
调用获取播放歌曲接口
把获取到的播放地址，通过$parent传递给父组件

### 05.推荐MV

在created生命周期函数中调用推荐MV接口
获取到数据并渲染到页面.上

## 推荐歌单

axios已经下载过，我们只需要导入即可使用

### 01.精品歌单

在created调用推荐歌单接口
获取到数据之后渲染到页面上

### 02.歌单列表

在created调用歌单列表接口
获取到数据之后渲染到页面上
播放量的信息，需要处理一下，如果超过了10万就显示xx万

### 03.切换分类

点击分类时，高亮当前分类
顶部数据切换，底部数据切换
在数据改变的时候要执行逻辑，可以使用侦听器watch来实现

### 04.分页效果

歌单列表获取到之后，设置总条数给分页组件
为分页组件绑定页码和页容量，以便进行联动,在调用接口时，-起传递
切换分类时，获取第一页的数据
分页组件页码切换时，获取对应页的数据.

## 最新音乐

axios已经下载过,我们只需要导入即可使用

### 01.歌曲列表

在created调用最新音乐接口
获取到数据之后渲染到页面上
歌曲的时间默认的是毫秒数，我们需要处理一-下，变成分秒的形式

### 02.分类切换

点击分类时，高亮当前分类
顶部数据切换，底部数据切换
在数据改变的时候要执行逻辑，可以使用侦听器watch来实现

### 03.点击播放

为播放按钮绑定点击事件
调用获取播放歌曲接口
把获取到的播放地址，通过$parent传递给父组件

## 最新MV

axios已经下载过，我们只需要导入即可使用

### 01.MV列表

在created调用全部MV接口
获取到数据之后渲染到页面上，先不考虑分类，和分页
播放量的信息，需要处理一下，如果超过了10万就显示xx万

### 02.分类切换

点击分类时，高亮当前分类.
顶部数据切换，底部数据切换
在数据改变的时候要执行逻辑，可以使用侦听器watch来实现，多写几个分类即可

### 03.数据分页

MV数据获取到之后，设置总条数给分页组件
为分页组件绑定页码和页容量，以便进行联动,在调用接口时，一起传递
切换分类时，获取第一页的数据
分页组件页码切换时，获取对应页的数据

## 搜索

组件名叫: 05.result.vue
路由地址是: /result

### 01.页面跳转

从顶部的搜索栏搜索跳转.
携带数据去往搜索结果(/result)页
把搜索的关键字渲染到顶部

### 02.歌曲搜索

导入axios
在created生命周期函数中调用搜索接口
默认先查询歌曲类型
把获取到的数据渲染到页面_上，顶部的数量别忘了
mv图标和歌曲时长需要处理一下
这里没有单独的播放按钮,
我们双击播放

### 03.歌单搜索

在切换到歌单分类的时候，搜索对应的数据，这里使用watch实现
获取到数据并渲染到页面上，播放次数超过10W特殊处理一下

### 04.MV搜索

在切换到MV分类的时候，搜索对应的数据，这里使用watch实现
为了页面布局美观，在获取MV时，个数可以稍微调整一下

## 歌单详情

歌单详情多个地方都可以跳转，记得在跳转的时候携带数据
组件名叫: 06.playlist.vue
路由地址是: /playlist

### 01.歌单信息

在created调用歌单信息接口
获取到数据之后渲染到页面上

### 02.热门评论

在created调用热门评论接口
获取到数据之后渲染到页面上，回复的评论信息注意一下即可

### 03.最新评论

在created调用最新评论接口
获取到数据之后渲染到页面上
把分页需要的数据和分页组件进行联动就好，和之前的做法类似

## mv详情

MV详情多个地方都可以跳转，记得在跳转的时候携带数据
组件名叫: 07.mv.vue
路由地址是: /mv

### 01.播放MV

在created调用MV地址接口
获取到数据之后把MV的地址设置给video标签即可

### 02.相关MV

在created调用相关MV接口
获取到数据之后把数据渲染到页面上即可

### 03.MV信息

在created调用MV信息接口
获取到MV数据之后把数据渲染到页面.上
在MV信息获取到之后，
调用歌手信息接口
获取到歌手信息之后把数据渲染到对应的位置

### 04.评论信息

在created调用MV评论接口
获取到评论信息之后，分别设置给热门评论以及最新评论即可
把分页需要的数据和分页组件进行联动就好，和之前的做法类似

## props & emit

弄清楚谁是父组件谁是子组件
props在子组件中设置，作用是定义允许父组件传入的值
emit在子组件中触发，通过他可以触发父组件中注册的事件，甚至传递参数

# guoguomusic播放器实现

webplayer有全部的代码，自己梳理一下结构然后实现一下。

## 1. 入口main.js

启动是npm run serve

然后入口是main.js

Vue element router css都是从这里配置的

## 2. App.vue

顶级组件App.vue，放了两个组件top.vue index.vue

先把所有的vue页面都重命名，然后自己从html页面开始搭建。

只先把结构列出来，然后再写js

## 3. 用到了element-ui得先导入进来

在main.js里引入，然后具体的内容在import element from '@/utils/element'里

## 4. top.vue

![image-20210119170619173](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210120142053.png)

布完局了，element-ui也引入了，然后下一步输入的数据得存起来吧。

去js export default {} 定义变量，然后写搜索，按键等一系列js代码。

## 5. index.vue

![image-20210119171608149](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20210120142053.png)

左边的nav栏一用router-view包起来立马样式就变好看然后居中了。

中间的main就放了一个router-view

下边的player就放了一个controls控件

用上router路由瞬间单页面应用的感觉就出来了！nice！

因为跳转到了不同的页面，所以该写分别的具体的跳转之后的页面了。



6 7 8 9 四个页面都当做组件，通过router-view的形式引入到main.js中了。

得先import，然后再配置path

## 6. discovery.vue

去写发现音乐的时候需要用到axios，是单独抽离到api/discovery.js下了，而它用引用了utils/request.js

request.js就先复制粘贴过来吧。

然后api/discovery.js是通过调用axios拿到接口的数据

import到discovery.vue里，通过created, v-for渲染数据到页面上，通过绑定@click方法，写js函数实现播放和跳转的方法。

剩下的方法大同小异，还剩下一个功能就是搜索结果跳到具体的结果页面，所以下一个我要写的页面是result.vue！

## 7. result.vue

接下来页面的步骤基本都遵循了以下的步骤：

1.  api/xxx.js拿到数据
2.  xxx.vue实现布局
3.  v-for渲染数据
4.  js实现方法

昨天晚上写完了，接下去写playlists.vue

## 8. playlist.vue

接下来页面的步骤基本都遵循了以下的步骤：

1.  api/xxx.js拿到数据
2.  xxx.vue实现布局
3.  v-for渲染数据
4.  js实现方法

然后点进去是一个个具体的歌单，所以下一个页面需要写的是playlist.vue

## 9. playlist.vue

接下来页面的步骤基本都遵循了以下的步骤：

1.  api/xxx.js拿到数据
2.  xxx.vue实现布局
3.  v-for渲染数据
4.  js实现方法

直接复制粘贴过去了。

## 10. songs.vue

这个也是直接粘贴过去了。

但是一页上直接显示100条数据，加载过慢，而且不好找，自己加一个分页的功能。

还不能加分页的功能，因为接口文件只有一个type类型参数，好吧🤣，那我把分页的代码取消了。🤣

## 11. mvs.vue

直接复制粘贴xxx.js   xxx.vue

## 12. mv.vue

直接复制粘贴xxx.js   xxx.vue

# webplayer上线

上线得买服务器。阿里云学生3次认证失败，一年内不能学生认证了😭。

Over，上传到GitHub上好了。

# [上传到GitHub](https://zhuanlan.zhihu.com/p/34625448)

**第一步：建立git仓库，cd到你的本地项目根目录下，执行git命令**

```text
git init
```

**第二步：将项目的所有文件添加到仓库中**

```c
git add .
```

**第三步：将add的文件commit到仓库**

```text
git commit -m "注释语句"
```

**第四步：去github上创建自己的Repository**

点击**Clone or download**按钮，复制弹出的地址**[git@github.com](mailto:git@github.com):\**\*/test.git**，记得要用SSH的地址，尽量不要用HTTPS的地址，如上图所示

**第五步：将本地的仓库关联到github上---把上一步复制的地址放到下面**

```text
git remote add origin git@github.com:***/test.git
```

**第六步：上传github之前，要先pull一下，执行如下命令：**

```text
git pull origin master
```

**第七步，上传代码到github远程仓库**

```text
git push -u origin master
```

还剩下修改README.md文件。

# 知识点

##1. playlists.vue里数据联动是怎么实现的？

##2. 数据分页是如何实现的？

## 小问题？

### 待解决

top.vue中import top from '@/components/top';  @是什么意思？

@keyup.enter.native的native是什么作用？

<i slot="prefix" 这个slot一写上去就有搜索🔍小图标了为什么？

为什么Vue启动后，手机也能通过本地局域网访问成功？

### 已解决

全局的样式是从哪里导入的？ Main.js





