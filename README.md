## 密集人群运动分析

组长：晏瑞阳

组员：蔡烨南 王佳豪 王亦飞 丁心成

### 代码仓库介绍
```
├── Dataset
│   ├── Dataset.py      数据加载类
│   └── _init_.py
├── README.md
├── core
│   ├── _init_.py
│   ├── dataloader.py   数据加载管理
│   ├── loss.py         loss函数管理
│   ├── model.py        模型管理
│   ├── optimizer.py
│   └── utils.py        常用工具
├── evaluate.py
├── main.py             可运行主函数
├── models
│   └── crowdflow.py
├── opts.py             参数管理
├── result
│   ├── checkpoints
│   ├── logs
│   ├── models
│   └── tensorboard     tensorboard可视化结果
├── tools               工具
│   └── check.py        
└── train.py
```
### 小组进度
详细进度可在`每周汇总`文件夹中查看  
进度简述：
1. 已完成对基于光流的RAFT方法的复现，正在进行对多目标跟踪或运动滤波方法的代码复现；
2. 预计在第三周（5.25前）完成RAFT方法的综述终稿，以及其他四篇参考论文的综述草稿；
3. WuhanMetro数据集方面，作为改进目标的baseline（RAFT）模型无需额外标注数据集，其他方法可能有不同的标注需要，但目前正处于公开数据集测试阶段。
4. RAFT的代码迁移和demo制作工作也已经开始
