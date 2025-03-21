# OSPF拥塞控制系统项目结构
satellite_congestion_control/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── ospf_router.py            # OSPF路由实现
│   ├── lsa_manager.py            # 链路状态通告(LSA)管理
│   ├── congestion_detector.py    # 拥塞检测器
│   └── packet.py                 # 数据包定义
├── models/
│   ├── __init__.py
│   ├── satellite.py              # 卫星节点模型
│   └── link.py                   # 链路模型，包含OSPF成本计算
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # 性能指标计算
│   └── config.py                 # 配置参数
└── main.py                       # 主程序入口
