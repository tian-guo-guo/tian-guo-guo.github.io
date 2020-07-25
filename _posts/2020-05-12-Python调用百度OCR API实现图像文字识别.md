---
layout:     post                    # 使用的布局（不需要改）
title:      Python调用百度OCR API实现图像文字识别              # 标题 
subtitle:   Python调用百度OCR API实现图像文字识别 #副标题
date:       2020-05-12              # 时间
author:     甜果果                      # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-keybord.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - python
    - 技术

---

# Python调用百度OCR API实现图像文字识别

百度[文字识别](https://cloud.tencent.com/product/ocr?from=10680)OCR接口提供了自然场景下整图文字检测、定位、识别等功能。文字识别的结果可以用于翻译、搜索、验证码等代替用户输入的场景。

![image-20200725163711422](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200725163711.png)



支持Python版本：2.7.+ ,3.+

首先安装接口模块，在电脑终端里执行 pip install baidu-aip 即可。

```python
from aip import AipOcr
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
def get_OCR_msg(name):
	with open(name + '.png','rb') as f:
		# reserved = open("reserved.txt", 'a')
        
		img = f.read()
		msg = client.basicGeneral(img)
		result_para = []
		for i in msg.get('words_result'):
			result = i.get('words')
			# reserved.write(result)
            
			# reserved.write("\n")
            
			result_para.append(result)
			print(result)
		# print(''.join(result_para))

if __name__ == "__main__":
	for i in range(1, 14):
		name = "screenshot "+str(i)
		get_OCR_msg(name)
```

APP_ID 、API_KEY、SECRET_KEY 三个值对应在http://console.bce.baidu.com/ai/#/ai/ocr/app/list 这里找到，需要用百度账号登录，然后创建一个应用

![image-20200725163838355](https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200725163838.png)

这样即可完成调用。