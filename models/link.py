# models/link.py
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import time
import numpy as np


class Link:
    """卫星链路模型"""

    def __init__(self, source_id: Tuple[int, int], target_id: Tuple[int, int],
                 capacity: float = 25.0, queue_size: int = 100,
                 distance: float = 1.0, link_cost_alpha: float = 0.5):
        """
        初始化链路

        Args:
            source_id: 源卫星ID (轨道面索引, 卫星索引)
            target_id: 目标卫星ID (轨道面索引, 卫星索引)
            capacity: 链路容量(Mbps)
            queue_size: 队列大小(数据包数量)
            distance: 链路传输距离(相对单位)
            link_cost_alpha: 链路成本缓存影响因子
        """
        self.source_id = source_id
        self.target_id = target_id
        self.capacity = capacity * 1024 * 1024  # Mbps转换为bps
        self.queue_size = queue_size
        self.queue = []
        self.distance = distance
        self.link_cost_alpha = link_cost_alpha

        # 性能指标相关
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.processed_bytes = 0
        self.total_packets = 0
        self.dropped_packets = 0
        self.queue_history = []

    @property
    def queue_occupancy(self) -> float:
        """
        计算队列占用率

        Returns:
            float: 队列占用率 (0.0-1.0)
        """
        return len(self.queue) / self.queue_size

    def calculate_ospf_cost(self) -> float:
        """
        计算OSPF链路成本, 根据公式: Cost = dis(1 + αx)

        Returns:
            float: OSPF链路成本
        """
        x = self.queue_occupancy  # 队列占用率
        cost = self.distance * (1 + self.link_cost_alpha * x)
        return cost

    def enqueue(self, packet: 'DataPacket') -> bool:
        """
        数据包入队

        Args:
            packet: 数据包

        Returns:
            bool: 如果入队成功返回True，否则返回False
        """
        self.total_packets += 1

        if len(self.queue) >= self.queue_size:
            self.dropped_packets += 1
            return False

        self.queue.append(packet)
        return True

    def dequeue(self) -> Optional['DataPacket']:
        """
        数据包出队，考虑链路容量和处理时间

        Returns:
            Optional[DataPacket]: 出队的数据包，如果队列为空返回None
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time

        # 计算这段时间内能处理的数据量
        processable_bytes = int(self.capacity * elapsed_time / 8)  # 转换为字节

        if self.queue and self.processed_bytes <= processable_bytes:
            packet = self.queue.pop(0)
            self.processed_bytes += packet.size

            # 如果已处理数据量超过阈值，更新时间和计数
            if self.processed_bytes >= processable_bytes:
                self.last_update_time = current_time
                self.processed_bytes = 0

            return packet

        return None

    def update_queue_history(self):
        """更新队列历史记录"""
        current_time = time.time()
        self.queue_history.append((current_time - self.last_update_time, len(self.queue)))
        self.last_update_time = current_time

    def get_packet_loss_rate(self) -> float:
        """计算丢包率"""
        if self.total_packets == 0:
            return 0.0
        return self.dropped_packets / self.total_packets

    def get_average_queue_length(self) -> float:
        """计算平均队列长度"""
        if not self.queue_history:
            return 0.0
        return sum(length for _, length in self.queue_history) / len(self.queue_history)

    def get_link_id(self, direction: str) -> str:
        """
        获取链路ID

        Args:
            direction: 链路方向 (north, south, east, west)

        Returns:
            str: 链路ID
        """
        return f"S{self.source_id[0]}-{self.source_id[1]}-{direction}"

    def update_metrics(self, metrics):
        """
        更新性能指标

        Args:
            metrics: 性能指标对象
        """
        from core.congestion_detector import QueueStateUpdatePacket

        # 修改链路ID的格式为统一格式
        link_id = f"S{self.source_id[0]}-{self.source_id[1]}-S{self.target_id[0]}-{self.target_id[1]}"

        qsup = QueueStateUpdatePacket(
            link_id=link_id,
            queue_occupancy=self.queue_occupancy
        )

        if hasattr(metrics, 'process_qsup'):
            metrics.process_qsup(qsup)