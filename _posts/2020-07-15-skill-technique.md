---
layout:     post           # 使用的布局（不需要改）
title:      skill-technique
subtitle:   skill-techniquen  #副标题
date:       2020-07-15             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
 
---

# Skill-technique

## 1. [文件相关](https://www.jianshu.com/p/d5030db5da0e)

**查看文件**

```bash
$ file test
test: UTF-8 Unicode text
```

**转换**

```bash
$ iconv -f utf8 -t gbk test -o test.gbk
```

**效果**

```bash
$ file test*
test:          UTF-8 Unicode text
test.gbk:     ISO-8859 text
```

## 2. 音视频相关

### flv2mp4

```
ffmpeg -i inputfile.flv -c copy outputfile.mp4
```

### 批量flv2mp4

```

```

### m4a2mp3

```
ffmpeg -i input.m4a output.mp3
```

### 下载youtube视频: [youtubemy](https://www.youtubemy.com/)

```
https://www.youtubemy.com/
```

### 批量下载bilibili视频

```bash
you-get --playlist -o /Volumes/NETAC/教学视频/黑马前端就业班(pink老师)第五部分/ --format=flv https://www.bilibili.com/video/av626270590
```

注意，网址必须是av号开头的，可以[检查网页源代码](https://www.jianshu.com/p/ca964ee86ffc)找到。复制head中内容到sublime，control+F
`https://www.bilibili.com/video/av`即可找到av号。

### 使用PYTHON you-get下载bilibili视频

```python
import sys
from you_get import common as you_get
url= 'https://www.bilibili.com/video/BV145411K7K3'
sys.argv=['you-get','-i',url]
you_get.main()
```



## 3. word

### word 一保存图片质量就下降怎么办

打开word，找到“文件”选项。 在左侧边栏中找到“选项”并点击。 选择“高级”，进入word的高级设置。 下拉找到关于“图像大小和质量”的内容，勾选“不压缩文件中的图像”，然后在重新插入高质量的图片，这样图片模糊的问题就解决了。