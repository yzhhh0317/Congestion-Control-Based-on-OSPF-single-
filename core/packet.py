# core/packet.py
from dataclasses import dataclass
from typing import Tuple, List, Optional
import time
import numpy as np


@dataclass
class DataPacket:
    """数据包类"""

    id: int  # 唯一标识符
    source: Tuple[int, int]  # 源节点网格坐标
    destination: Tuple[int, int]  # 目标节点网格坐标
    size: int = 1024 * 8  # 数据包大小(bits)
    creation_time: float = None  # 创建时间

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.time()


@dataclass
class LSAPacket:
    """链路状态通告包"""

    link_id: str  # 链路标识符
    cost: float  # 链路成本
    source_id: Tuple[int, int]  # 源卫星ID
    sequence_number: int  # 序列号
    timestamp: float = None  # 时间戳

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TrafficGenerator:
    """流量生成器"""

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
            'normal': 0.3,  # 正常状态：30%容量
            'warning': 0.5,  # 预警状态：50%容量
            'congestion': 0.7,  # 拥塞状态：70%容量
        }

    def calculate_packets_per_step(self, state: str) -> int:
        """
        计算每个时间步应生成的数据包数量

        Args:
            state: 当前链路状态

        Returns:
            int: 数据包数量
        """
        state = state if state in self.state_ratios else 'normal'
        target_rate = self.link_capacity * self.state_ratios[state]

        # 计算理论包数
        packets_per_step = (target_rate * self.time_step) / self.packet_size

        # 添加随机扰动
        actual_packets = int(packets_per_step * (1 + np.random.uniform(-0.1, 0.1)))

        # 保证至少生成一些包，但不要太多
        return max(1, min(actual_packets, 50))

    def generate_packets(self, source: Tuple[int, int], state: str,
                         num_planes: int, sats_per_plane: int) -> List[DataPacket]:
        """
        生成一组数据包

        Args:
            source: 源节点坐标
            state: 当前状态
            num_planes: 轨道面数量
            sats_per_plane: 每个轨道面的卫星数量

        Returns:
            List[DataPacket]: 生成的数据包列表
        """
        packets = []
        num_packets = self.calculate_packets_per_step(state)

        for _ in range(num_packets):
            # 随机选择目标卫星
            dest_i = np.random.randint(0, num_planes)  # 轨道面
            dest_j = np.random.randint(0, sats_per_plane)  # 轨道内编号
            destination = (dest_i, dest_j)

            # 确保目标不是源节点
            if destination == source:
                continue

            packet = DataPacket(
                id=int(time.time() * 1000000),  # 微秒级时间戳作为ID
                source=source,
                destination=destination
            )
            packets.append(packet)

        return packets

    def generate_hotspot_traffic(self, hotspot_sources: List[Tuple[int, int]],
                                 ground_stations: List[Tuple[int, int]],
                                 state: str) -> List[DataPacket]:
        """
        生成热点区域流量

        Args:
            hotspot_sources: 热点区域的卫星坐标列表
            ground_stations: 地面站坐标列表
            state: 当前状态

        Returns:
            List[DataPacket]: 生成的数据包列表
        """
        packets = []

        # 为每个热点源生成流量
        for source in hotspot_sources:
            # 大部分流量发往地面站
            ground_station_ratio = 0.7

            num_packets = self.calculate_packets_per_step(state) * 2  # 热点流量翻倍

            for _ in range(num_packets):
                if np.random.random() < ground_station_ratio and ground_stations:
                    # 发往地面站
                    destination = ground_stations[np.random.randint(0, len(ground_stations))]
                else:
                    # 发往随机卫星
                    destination = hotspot_sources[np.random.randint(0, len(hotspot_sources))]
                    while destination == source:  # 确保不是自己
                        destination = hotspot_sources[np.random.randint(0, len(hotspot_sources))]

                packet = DataPacket(
                    id=int(time.time() * 1000000),  # 微秒级时间戳作为ID
                    source=source,
                    destination=destination
                )
                packets.append(packet)

        return packets