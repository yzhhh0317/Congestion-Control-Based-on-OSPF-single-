# models/link.py
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import time
import numpy as np


class Link:
    """
    星间链路模型

    处理数据包的入队、出队和队列状态管理
    适应分布式概率拥塞控制算法需求
    """

    def __init__(self, source_id: Tuple[int, int], target_id: Tuple[int, int],
                 capacity: float = 25.0, queue_size: int = 200):
        """
        初始化链路

        Args:
            source_id: 源节点ID (网格坐标)
            target_id: 目标节点ID (网格坐标)
            capacity: 链路容量(Mbps)
            queue_size: 队列最大容量(数据包数量)
        """
        self.source_id = source_id
        self.target_id = target_id
        self.capacity = capacity * 1024 * 1024  # Mbps转换为bps
        self.queue_size = queue_size
        self.queue = []
        self.last_update_time = time.time()
        self.processed_bytes = 0
        self.start_time = time.time()

        # 性能统计
        self.total_packets = 0
        self.dropped_packets = 0
        self.queue_history = []

        # 流量度量相关
        self.last_metric_update = time.time()
        self.metric_update_interval = 0.1  # 流量度量更新间隔(秒)

    @property
    def queue_occupancy(self) -> float:
        """
        计算队列占用率

        Returns:
            float: 队列占用率(0-1)
        """
        return len(self.queue) / self.queue_size if self.queue_size > 0 else 0

    @property
    def direction(self) -> str:
        """
        根据源节点和目标节点确定链路方向

        Returns:
            str: 链路方向('north', 'south', 'east', 'west')
        """
        src_i, src_j = self.source_id
        tgt_i, tgt_j = self.target_id

        if src_i < tgt_i:
            return 'east'
        elif src_i > tgt_i:
            return 'west'
        elif src_j < tgt_j:
            return 'south'
        elif src_j > tgt_j:
            return 'north'
        else:
            return 'unknown'

    def enqueue(self, packet) -> bool:
        """
        数据包入队

        Args:
            packet: 数据包对象

        Returns:
            bool: 入队是否成功
        """
        self.total_packets += 1

        # 队列已满，丢弃数据包
        if len(self.queue) >= self.queue_size:
            self.dropped_packets += 1
            return False

        # 数据包入队
        self.queue.append(packet)
        return True

    def dequeue(self) -> Optional['DataPacket']:
        """
        数据包出队，考虑链路带宽限制

        Returns:
            DataPacket 或 None: 出队的数据包，如果队列为空或未达处理时间则返回None
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

    def get_queue_state(self) -> Dict:
        """
        获取队列状态

        Returns:
            dict: 包含队列状态信息的字典
        """
        return {
            'queue_length': len(self.queue),
            'queue_occupancy': self.queue_occupancy,
            'queue_capacity': self.queue_size,
            'dropped_packets': self.dropped_packets,
            'total_packets': self.total_packets,
            'loss_rate': self.get_packet_loss_rate()
        }

    def update_queue_history(self):
        """更新队列历史记录，用于性能分析"""
        current_time = time.time()
        self.queue_history.append((current_time - self.last_update_time, len(self.queue)))
        self.last_update_time = current_time

    def get_packet_loss_rate(self) -> float:
        """
        计算丢包率

        Returns:
            float: 丢包率(0-1)
        """
        if self.total_packets == 0:
            return 0.0
        return self.dropped_packets / self.total_packets

    def get_average_queue_length(self) -> float:
        """
        计算平均队列长度

        Returns:
            float: 平均队列长度
        """
        if not self.queue_history:
            return 0.0

        # 计算加权平均队列长度
        total_time = 0
        weighted_sum = 0

        for time_interval, queue_length in self.queue_history:
            total_time += time_interval
            weighted_sum += queue_length * time_interval

        return weighted_sum / total_time if total_time > 0 else 0

    def should_update_metrics(self) -> bool:
        """
        判断是否应该更新流量度量

        Returns:
            bool: 是否应该更新
        """
        current_time = time.time()
        if current_time - self.last_metric_update >= self.metric_update_interval:
            self.last_metric_update = current_time
            return True
        return False