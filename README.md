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
### 小组进度（截止至6.5）
详细进度可在`每周汇总`文件夹中查看  
进度简述：
1. 已完成对基于光流的RAFT方法的复现和对多目标跟踪或运动滤波方法的代码复现以及代码迁移工作
2. 已完成五篇论文的综述终稿，详情可见`论文综述`
3. 开展小股人流数据集收集工作并进行预处理，该工作为常态任务，每周两次拍摄。
4. 正在进行SocialVAE、SHENet的复现工作
