---
layout:     post           # 使用的布局（不需要改）
title:      利用Jenkins + nginx 实现前端项目自动构建与持续集成           # 标题 
subtitle:   利用Jenkins + nginx 实现前端项目自动构建与持续集成 #副标题
date:       2020-10-14             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/home-bg-art.jpg    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 前端
    - 面试
---

# [利用Jenkins + nginx 实现前端项目自动构建与持续集成](https://juejin.im/post/6844903845936496654)

#### 实现目标

本地push代码到GitHub，Webhook自动触发jenkins上的构建动作,完成安装node插件并且打包，然后通过Publish Over SSH插件，将打包出来的文件，部署到目标服务器上。

### 前期准备

1.  github 账号和项目
2.  centos 服务器;
3.  服务器安装 Java SDK;
4.  服务器安装 nginx + 启动；
5.  服务器安装jenkins + 启动；

### jenkins介绍

Jenkins是开源的,使用Java编写的持续集成的工具，在Centos上可以通过yum命令行直接安装。Jenkins只是一个平台，真正运作的都是插件。这就是jenkins流行的原因，因为jenkins什么插件都有。

#### 首先登录服务器更新系统软件

```
$ yum update
复制代码
```

#### 安装Java和git

```
$ yum install java
$ yum install git
复制代码
```

#### 安装nginx

```
$ yum install nginx //安装
$ service nginx start //启动
复制代码
```

出现Redirecting to /bin/systemctl start  nginx.service

说明nginx已经启动成功了，访问http://你的ip/，如果成功安装会出来nginx默认的欢迎界面

![006tNc79gy1g344jsd94xj32700s6tdt.jpg](https://user-gold-cdn.xitu.io/2019/5/17/16ac471b8be0e05f?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)



#### 安装Jenkins

```
$ wget -O /etc/yum.repos.d/jenkins.repo http://pkg.jenkins-ci.org/redhat/jenkins.repo
rpm --import https://jenkins-ci.org/redhat/jenkins-ci.org.key 

$ yum install jenkins //完成之后直接使用 yum 命令安装 Jenkins

$ service jenkins restart  //启动 jenkins
复制代码
```

jenkins启动成功后默认的是8080端口，浏览器输入你的服务器 ip 地址加8080 端口就可以访问了。



![006tNc79gy1g34500gsipj31id0u0thy.jpg](https://user-gold-cdn.xitu.io/2019/5/17/16ac4715b68352cc?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)



输入 cat /var/lib/jenkins/secrets/initialAdminPassword 查看初始密码

这里我们选择推荐通用插件安装即可，选择后等待完成插件安装以及初始化账户

![WX20190517-111347@2x.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471ef76433d7?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

![WX20190517-111420@2x.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471b8c0ee387?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

![WX20190517-111734@2x.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471b8c2afade?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)



然后安装两个推荐的插件 [Rebuilder](https://wiki.jenkins.io/display/JENKINS/Rebuild+Plugin) [SafeRestart](https://wiki.jenkins.io/display/JENKINS/SafeRestart+Plugin)

### 在jenkins中安装nodeJs插件

因为我们的项目是要用到node打包的，所以先在jenkins中安装nodeJs插件，安装后进入全局工具配置，配置一个我们要用到的node版本。

![9999.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471b8c1aa34d?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

![9090.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471b8c01e213?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)



### 创建任务

1.  点击创建一个新任务

    ![45.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac4716f16fa79c?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

    ![67.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac4716f78376ff?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

    

2.  jenkins关联 GitHub项目地址

    ![89.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1280" height="805"></svg>)

    

3.  选择构建环境并编写shell 命令

    ![90999999.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1280" height="877"></svg>)

    

配置完成后点击立即构建，等待构建完，点击工作空间，可以发现已经多出一个打包后的dist目录。点击控制台输出可以查看详细构建log

![123.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac47176d7156da?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

![45.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471a1340e15d?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

![78.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471a15c8cf73?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)



到这里已经实现了本地代码提交到github，然后在jenkins上点击构建，可以拉取代码并且打包，下一步实现打包后的dist目录放到目标服务器上。

### 安装Publish Over SSH 插件，我们将通过这个工具实现服务器部署功能。

安装完成后在系统管理-> 系统设置->Publish over SSH 里设置服务器信息

```
Passphrase：密码（key的密码，没设置就是空）
Path to key：key文件（私钥）的路径
Key：将私钥复制到这个框中(path to key和key写一个即可)

SSH Servers的配置：
SSH Server Name：标识的名字（随便你取什么）
Hostname：需要连接ssh的主机名或ip地址（建议ip）
Username：用户名
Remote Directory：远程目录（上面第二步建的testjenkins文件夹的路径）

高级配置：
Use password authentication, or use a different key：勾选这个可以使用密码登录，不想配ssh的可以用这个先试试
Passphrase / Password：密码登录模式的密码
Port：端口（默认22）
Timeout (ms)：超时时间（毫秒）默认300000

复制代码
```

这里配置的是账号密码登录，填写完后点击test，出现Success说明配置成功

![909090.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1219" height="1280"></svg>)



在刚才的testJenkins工程中配置**构建后操作**，选择send build artificial over SSH， 参数说明：

```
Name:选择一个你配好的ssh服务器
Source files ：写你要传输的文件路径
Remove prefix ：要去掉的前缀，不写远程服务器的目录结构将和Source files写的一致
Remote directory ：写你要部署在远程服务器的那个目录地址下，不写就是SSH Servers配置里默认远程目录
Exec command ：传输完了要执行的命令，我这里执行了进入test目录,解压缩,解压缩完成后删除压缩包三个命令

复制代码
```

注意在构建中添加压缩dist目录命令

![232323.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1280" height="899"></svg>)



填完后执行构建。成功后登录我们目标服务器发现test目录下有了要运行的文件

![565645.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1130" height="838"></svg>)



访问域名发现项目可以访问了

![4545.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="742" height="1280"></svg>)



接下来实现开发本地push代码到github上后，触发Webhook，jenkins自动执行构建。

1.  jenkins安装Generic Webhook Trigger 插件
2.  github添加触发器

### 配置方法

1.在刚才的testJenkins工程中点击构建触发器中选择Generic Webhook Trigger，填写token

![877777.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1280" height="927"></svg>)



2.github配置Webhook 选择github项目中的Settings->Webhooks>add webhook 配置方式按上图红框中的格式，选择在push代码时触发webhook，成功后会在下方出现一个绿色的小勾勾

![888888.png](data:image/svg+xml;utf8,<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1280" height="1055"></svg>)



测试一下，把vue项目首页的9900去了，然后push代码去github，发现Jenkins中的构建已经自动执行，

![0000.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471c2c30ecb7?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

查看页面也是ok的

![9843.png](https://user-gold-cdn.xitu.io/2019/5/17/16ac471ad4dfb88c?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)



一套简单的前端自动化工作流就搭建完成，是选择代码push后在Jenkins中手动构建，还是push后自动构建，看公司情况使用。



[Link](https://juejin.im/post/6844903845936496654)

