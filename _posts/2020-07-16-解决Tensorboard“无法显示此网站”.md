---
layout:     post           # 使用的布局（不需要改）
title:      vue踩坑-error 'res' is assigned a value but never used no-unused-vars
subtitle:   vue踩坑-error 'res' is assigned a value but never used no-unused-vars  #副标题
date:       2020-12-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - vue
 
---

# vue踩坑-error 'res' is assigned a value but never used no-unused-vars

写代码的时候，遇到了这样的一个报错

```
error  in ./src/views/CategoryEdit.vue
Module Error (from ./node_modules/eslint-loader/index.js):
D:\node-vue-moba\admin\src\views\CategoryEdit.vue
24:11  error  'res' is assigned a value but never used  no-unused-vars
✖ 1 problem (1 error, 0 warnings)
✖ 1 problem (1 error, 0 warnings)
 @ ./src/router/index.js 4:0-53 13:15-27
 @ ./src/main.js
 @ multi (webpack)-dev-server/client?http://192.168.0.121:8080/sockjs-node (webpack)/hot/dev-server.js ./src/main.js
```

![5640239-a2cdd53a58bf46ac.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201214211510.png)

错误原因：eslint的验证语法

解决办法：在错误语句后添加注释

```
// eslint-disable-line no-unused-vars
```

代码如下所示

```
methods: {
  async  save(){
    const res = await this.$http.post('categories',this.model) // eslint-disable-line no-unused-vars
    this.$router.push('/categories/list')
    this.$message({
        type:'success',
        message:'保存成功'
    })
    }
},
```

这个时候就不报错了，页面就能看到了。



![5640239-4d1cf48a5f06601b.png](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20201214211510.png)