## 密集人群运动分析

组长：晏瑞阳

组员：蔡烨南 王佳豪 王亦飞 丁心成

### 小组进度（截止至6.9）
详细进度可在`每周汇总`文件夹中查看  
进度简述：
1. 已完成五篇论文的综述终稿，详情可见`论文综述`（RAFT，Deep_sort，SHENet，SocialVAE，CPEPF）；
2. 已完成RAFT、Deep_sort、SocialVAE三种方法的复现，并正在服务器上集成RAFT、Deep_sort两种方法并完成demo制作；
3. 已采集大约100个视频片段，计划在项目结束前采集200个视频并完成预处理等工作；
4. 正在研究如何改进RAFT光流法；
5. 正在进行SHENet的复现，但环境配置较为特殊，不便迁移；
6. 正在安排项目报告的撰写和答辩ppt的学习制作。

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

