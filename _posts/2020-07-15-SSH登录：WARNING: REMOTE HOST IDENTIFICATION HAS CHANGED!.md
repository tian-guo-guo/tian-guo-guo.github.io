---
layout:     post           # 使用的布局（不需要改）
title:      SSH登录WARNING REMOTE HOST IDENTIFICATION HAS CHANGED!
subtitle:   SSH WARNING REMOTE HOST IDENTIFICATION HAS CHANGED  #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
 
---

# SSH登录：WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!

服务器重装了以后，用VScode连接服务器的时候报错：

```
[14:12:24.825] Stopped parsing output early. Remaining text: OpenSSH_7.9p1, LibreSSL 2.7.3debug1: Server host key: ecdsa-sha2-nistp256 SHA256:+80U+jF934ijEiphQPWRw4mwlo1sjQ1Hco2WAUmLT3o@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!Someone could be eavesdropping on you right now (man-in-the-middle attack)!It is also possible that a host key has just been changed.The fingerprint for the ECDSA key sent by the remote host isSHA256:+80U+jF934ijEiphQPWRw4mwlo1sjQ1Hco2WAUmLT3o.Please contact your system administrator.Add correct host key in /Users/suntian/.ssh/known_hosts to get rid of this message.Offending ECDSA key in /Users/suntian/.ssh/known_hosts:9ECDSA host key for [211.82.97.237]:5722 has changed and you have requested strict checking.Host key verification failed.
[14:12:24.825] Failed to parse remote port from server output
[14:12:24.826] Resolver error: 
[14:12:24.831] ------
```

出现这个问题的原因是,第一次使用SSH连接时，会生成一个认证，储存在客户端的known_hosts中。而由于服务器重新安装系统了，所以会出现以上错误。

解决办法：

进入本机的/Users/suntian/.ssh/，打开known_hosts，删掉237 5722那一行

```
[211.82.97.237]:5722 ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFpdFuGiluftX3zRjdeowluWA0QIIAHba/qFsGnOTLV6eTFriH/rVLc85UTbqAnxzOTCvVdXvcK4BI84CgTUcSI=
[211.82.97.238]:5422 ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBBeRSayWWgnAYHmXRy1dKnNcJVw66zSACTs7wKM8eAJq6+SZMcf7DHfA0pQ59nola1A4rZzya3pO5OgsS3btRKY=
```

再次连接服务器会重新配置，即可连线成功。