# utils/config.py

"""
系统配置参数
为分布式概率拥塞控制算法提供配置支持
"""

SYSTEM_CONFIG = {
    # 星座配置
    'NUM_ORBIT_PLANES': 6,  # 轨道面数量
    'SATS_PER_PLANE': 11,  # 每个轨道面的卫星数量
    'POLAR_THRESHOLD': 75.0,  # 极地区域纬度阈值(度)

    # 链路参数
    'LINK_CAPACITY': 25,  # 链路容量(Mbps)
    'QUEUE_SIZE': 200,  # 队列大小(数据包数量)
    'PACKET_SIZE': 1024,  # 数据包大小(bytes)

    # DRA路由参数
    'DRA_DIRECTION_WEIGHT': 0.7,  # DRA方向权重(垂直vs水平)

    # 分布式概率拥塞控制参数
    'BUFFER_WEIGHT': 0.8,  # 本地缓冲区权重
    'NEIGHBOR_WEIGHT': 0.2,  # 邻居信息权重
    'PREF_PROBABILITY': 0.9,  # 主方向偏好概率
    'THRESHOLD': 150,  # 队列阈值
    'METRIC_UPDATE_INTERVAL': 0.1,  # 流量度量更新间隔(秒)
    'METRIC_VALIDITY_PERIOD': 1.0,  # 流量度量有效期(秒)

    # 拥塞检测参数
    'WARNING_THRESHOLD': 0.5,  # 预警阈值（队列占用率）
    'CONGESTION_THRESHOLD': 0.75,  # 拥塞阈值（队列占用率）
    'RELEASE_DURATION': 3,  # 拥塞解除持续时间(周期)

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
        'CONGESTION_DURATION': 15,  # 15s高强度流量
        'CONGESTION_INTERVAL': 60,  # 60s触发一次
        'TOTAL_DURATION': 240  # 240s总时长
    },

    # 仿真控制参数
    'SIMULATION_STEP': 0.01,  # 仿真步长(秒)
    'SIMULATION_SEED': 42,  # 随机数种子，确保结果可重现

    # 性能指标采集参数
    'METRICS_COLLECTION': {
        'SAMPLE_INTERVAL': 0.1,  # 采样间隔(秒)
        'AVERAGING_WINDOW': 10,  # 平均窗口大小
        'MIN_SAMPLES': 5,  # 最少采样数
        'REPORT_INTERVAL': 30  # 报告生成间隔(秒)
    },

    # 流量控制参数
    'TRAFFIC_CONTROL': {
        'BASE_RATIO': 0.45,  # 基础负载比例
        'WARNING_RATIO': 0.65,  # 预警阶段负载比例
        'PEAK_RATIO': 0.85,  # 高峰负载比例
        'VARIATION': 0.05  # 随机变化范围
    },

    # 可视化配置
    'VISUALIZATION': {
        'PLOT_DPI': 300,  # 图像DPI
        'PLOT_FIGSIZE': (10, 6),  # 图像尺寸
        'LINE_COLORS': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 线条颜色
    }
}