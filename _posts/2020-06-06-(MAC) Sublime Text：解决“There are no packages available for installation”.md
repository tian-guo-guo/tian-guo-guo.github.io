---
layout:     post           # 使用的布局（不需要改）
title:      (MAC) Sublime Text：解决“There are no packages available for installation”
subtitle:   sublime no packages avaliable error #副标题
date:       2020-06-06             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-ios9-web.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs

---

# (MAC) Sublime Text：解决“There are no packages available for installation”

解决方案
核心：
出现此问题的主要原因在于无法访问“https://packagecontrol.io/channel_v3.json”，因此只需要将其地址替换掉即可解决。

解决方案1：
打开Sublime Text，点击Sublime Text->Preferences->Settings - User，添加如下代码：

"channels":
[
"http://cst.stu.126.net/u/json/cms/channel_v3.json",
],
保存后即可解决。

解决方案2：
1、 Command+Shift+p 打开命令面板
2、输入 Package Control: Add Channel
3、添加地址：https://github.com/wilon/sublime/raw/master/download/channel_v3.json

参考贴：
[1] 解决sublime text无法安装插件问题
[2] Sublime Text 3 下载安装插件太慢怎么办?

作者：姚远_HIT
链接：https://www.jianshu.com/p/d059f5f1fe86
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。