satellite_congestion_control/
├── core/
│   ├── __init__.py
│   ├── dra_router.py                # 新增：DRA路由算法（方向估计和增强）
│   ├── probabilistic_controller.py  # 新增：分布式概率控制器
│   ├── congestion_detector.py       # 修改：简化拥塞检测逻辑
│   └── packet.py                    # 修改：增加流量度量信息字段
├── models/
│   ├── __init__.py
│   ├── satellite.py                 # 修改：添加流量信息存储
│   └── link.py                      # 保持不变
├── utils/
│   ├── __init__.py
│   ├── metrics.py                   # 修改：调整性能指标计算
│   └── config.py                    # 修改：更新配置参数
└── main.py                          # 修改：重构主逻辑