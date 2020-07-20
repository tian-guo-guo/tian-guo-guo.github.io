---
layout:     post           # 使用的布局（不需要改）
title:      ValueError Not a TBLoader or TBPlugin subclass <class  tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader >
subtitle:   tensorboard启动不成功  #副标题
date:       2020-07-16             # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@1.0/assets/img/post-bg-debug.png    #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 技术
    - bugs
    - 服务器
    - tensorflow
 
---

# ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>

模型训练完之后，想开tensorboard，看模型训练过程，输入`tensorboard --logdir=./output/ne_WIPO_NER`，结果报错了`ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>`，tensorboard没启动成功。

```
/root/miniconda3/envs/tensorflow/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
Traceback (most recent call last):
  File "/root/miniconda3/envs/tensorflow/bin/tensorboard", line 8, in <module>
    sys.exit(run_main())
  File "/root/miniconda3/envs/tensorflow/lib/python3.7/site-packages/tensorboard/main.py", line 59, in run_main
    program.get_default_assets_zip_provider())
  File "/root/miniconda3/envs/tensorflow/lib/python3.7/site-packages/tensorboard/program.py", line 144, in __init__
    self.plugin_loaders = [make_loader(p) for p in plugins]
  File "/root/miniconda3/envs/tensorflow/lib/python3.7/site-packages/tensorboard/program.py", line 144, in <listcomp>
    self.plugin_loaders = [make_loader(p) for p in plugins]
  File "/root/miniconda3/envs/tensorflow/lib/python3.7/site-packages/tensorboard/program.py", line 143, in make_loader
    raise ValueError("Not a TBLoader or TBPlugin subclass: %s" % plugin)
ValueError: Not a TBLoader or TBPlugin subclass: <class 'tensorboard_plugin_wit.wit_plugin_loader.WhatIfToolPluginLoader'>
```

查阅资料后得知，删掉tensorboard和tensorboard-plugin-wit就好了。

所以我就使用 `pip uninstall tensorboard `和 `pip uninstall tensorboard-plugin-wit `，搞定。

之前conda list

```
six                       1.15.0                   pypi_0    pypi
sqlite                    3.32.3               h62c20be_0  
tensorboard               1.14.0                   pypi_0    pypi
tensorboard-plugin-wit    1.7.0                    pypi_0    pypi
tensorflow                2.2.0                    pypi_0    pypi
tensorflow-estimator      1.14.0                   pypi_0    pypi
tensorflow-gpu            1.14.0                   pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
tk                        8.6.10               hbc83047_0  
```

删掉后是

```
six                       1.15.0                   pypi_0    pypi
sqlite                    3.32.3               h62c20be_0  
tensorflow                2.2.0                    pypi_0    pypi
tensorflow-estimator      1.14.0                   pypi_0    pypi
tensorflow-gpu            1.14.0                   pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
tk                        8.6.10               hbc83047_0  
```

[Link](https://www.pythonf.cn/read/99427)

