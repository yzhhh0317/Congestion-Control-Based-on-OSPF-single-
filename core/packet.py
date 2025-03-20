# core/packet.py
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import time
import numpy as np


@dataclass
class DataPacket:
    """
    数据包类

    在分布式概率拥塞控制算法中，数据包携带流量度量信息
    """
    id: int  # 唯一标识符
    source: Tuple[int, int]  # 源节点网格坐标
    destination: Tuple[int, int]  # 目标节点网格坐标
    size: int = 1024 * 8  # 数据包大小(bits)
    creation_time: float = None  # 创建时间
    traffic_metric: float = 0.0  # 流量度量信息

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.time()

    def get_age(self) -> float:
        """获取数据包年龄（秒）"""
        return time.time() - self.creation_time

    def update_traffic_metric(self, metric: float):
        """更新流量度量信息"""
        self.traffic_metric = metric


class TrafficGenerator:
    """
    流量生成器

    生成模拟网络负载的数据包
    """

    def __init__(self, link_capacity: float = 25.0):
        """
        初始化流量生成器

        Args:
            link_capacity: 链路容量(Mbps)
        """
        self.link_capacity = link_capacity * 1024 * 1024  # 转换为bps
        self.packet_size = 1024 * 8  # bits
        self.time_step = 0.01  # 时间步长(秒)

        # 状态对应的流量比例
        self.state_ratios = {
            'normal': 0.45,  # 正常状态：45%容量
            'warning': 0.65,  # 预警状态：65%容量
            'congestion': 0.85,  # 拥塞状态：85%容量
        }

    def calculate_packets_per_step(self, state: str) -> int:
        """
        计算每个时间步应生成的数据包数量

        Args:
            state: 链路状态('normal', 'warning', 'congestion')

        Returns:
            int: 数据包数量
        """
        state = state if state in self.state_ratios else 'normal'
        target_rate = self.link_capacity * self.state_ratios[state]

        # 计算理论包数
        packets_per_step = (target_rate * self.time_step) / self.packet_size

        # 添加随机扰动(±5%)
        actual_packets = int(packets_per_step * (1 + np.random.uniform(-0.05, 0.05)))

        # 确保至少有一些包，但不超过合理上限
        return max(1, min(actual_packets, 100))

    def generate_packets(self, source: Tuple[int, int], state: str,
                         num_orbit_planes: int, sats_per_plane: int) -> List[DataPacket]:
        """
        生成一组数据包

        Args:
            source: 源节点坐标
            state: 当前状态
            num_orbit_planes: 轨道面数量
            sats_per_plane: 每个轨道面的卫星数量

        Returns:
            list: 生成的数据包列表
        """
        packets = []
        num_packets = self.calculate_packets_per_step(state)

        for i in range(num_packets):
            # 随机选择目标卫星
            dest_i = np.random.randint(0, num_orbit_planes)
            dest_j = np.random.randint(0, sats_per_plane)
            destination = (dest_i, dest_j)

            # 确保目标不是源节点
            if destination == source:
                continue

            # 创建数据包
            packet = DataPacket(
                id=int(time.time() * 1000) + i,  # 毫秒时间戳加索引作为ID
                source=source,
                destination=destination
            )
            packets.append(packet)

        return packets


@dataclass
class TrafficMetricPacket:
    """
    流量度量包

    在分布式概率拥塞控制算法中，节点之间交换的流量度量信息
    """
    source_id: Tuple[int, int]  # 源节点ID
    direction: str  # 方向（'north', 'south', 'east', 'west'）
    queue_length: int  # 队列长度
    traffic_metric: float  # 流量度量值
    timestamp: float = None  # 时间戳

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class QueueStateUpdatePacket:
    """
    队列状态更新包

    用于节点间交换队列状态信息，支持分布式拥塞控制决策
    """
    source_id: Tuple[int, int]  # 源节点ID
    direction: str  # 方向
    queue_length: int  # 队列长度
    max_queue: int  # 最大队列长度
    timestamp: float = None  # 时间戳

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def queue_occupancy(self) -> float:
        """计算队列占用率"""
        return self.queue_length / self.max_queue if self.max_queue > 0 else 0