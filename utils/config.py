# utils/config.py
SYSTEM_CONFIG = {
    # 星座配置
    'NUM_ORBIT_PLANES': 6,  # 轨道面数量
    'SATS_PER_PLANE': 11,  # 每个轨道面的卫星数量

    # 链路参数
    'LINK_CAPACITY': 25,  # 链路容量(Mbps)
    'QUEUE_SIZE': 100,  # 队列大小(数据包数量)
    'PACKET_SIZE': 1024,  # 数据包大小(bytes)

    # OSPF拥塞控制参数
    'WARNING_THRESHOLD': 0.5,  # 预警阈值（队列占用率）
    'CONGESTION_THRESHOLD': 0.75,  # 拥塞阈值（队列占用率）
    'LINK_COST_ALPHA': 0.5,  # 链路成本缓存影响因子
    'CACHE_UPDATE_INTERVAL': 10.0,  # 缓存利用率更新间隔(秒)

    # LSA参数
    'LSA_UPDATE_INTERVAL': 10.0,  # LSA更新间隔(秒)
    'LSA_HOLD_TIME': 3.0,  # LSA保持时间(秒)

    # 拥塞场景配置
    'CONGESTION_SCENARIO': {
        'TYPE': 'single',  # 'single' 或 'multiple'
        'SINGLE_LINK': {
            'source_plane': 2,
            'source_index': 3,
            'direction': 'east'  # 第2轨道面第3颗卫星的东向链路
        },
        'MULTIPLE_LINKS': [
            {'source_plane': 2, 'source_index': 3, 'direction': 'east'},
            {'source_plane': 2, 'source_index': 4, 'direction': 'east'},
            {'source_plane': 3, 'source_index': 3, 'direction': 'east'},
            {'source_plane': 3, 'source_index': 4, 'direction': 'east'}
        ],
        'GROUND_STATIONS': [
            {'plane': 0, 'index': 5},
            {'plane': 2, 'index': 8},
            {'plane': 4, 'index': 2},
            {'plane': 5, 'index': 9}
        ],
        'HOTSPOTS': [
            {'plane': 1, 'index': 3},
            {'plane': 1, 'index': 4},
            {'plane': 2, 'index': 3},
            {'plane': 2, 'index': 4}
        ],
        'CONGESTION_DURATION': 15,  # 15s高强度流量
        'CONGESTION_INTERVAL': 60,  # 60s触发一次
        'TOTAL_DURATION': 240  # 240s总时长
    },

    # 仿真控制参数
    'SIMULATION_STEP': 0.01,  # 仿真步长(秒)

    # 性能指标采集参数
    'METRICS_COLLECTION': {
        'SAMPLE_INTERVAL': 0.1,  # 采样间隔
        'AVERAGING_WINDOW': 10,  # 平均窗口大小
        'MIN_SAMPLES': 5  # 最少采样数
    },

    # 流量控制参数
    'TRAFFIC_CONTROL': {
        'BASE_RATIO': 0.45,  # 基础负载比例
        'PEAK_RATIO': 0.85,  # 高峰负载比例
        'VARIATION': 0.05  # 随机变化范围
    }
}