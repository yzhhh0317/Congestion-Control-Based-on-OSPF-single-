# models/satellite.py
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Optional
import numpy as np
import time


@dataclass
class Satellite:
    """
    卫星节点类

    存储卫星位置、链路和流量度量信息
    支持分布式概率拥塞控制算法的决策机制
    """
    grid_pos: Tuple[int, int]  # 网格坐标 (i, j)
    is_monitored: bool = False  # 是否需要监控
    links: Dict[str, 'Link'] = field(default_factory=dict)  # 各方向链路
    traffic_metrics: Dict[str, float] = field(default_factory=dict)  # 各方向流量度量
    last_metrics_update: Dict[str, float] = field(default_factory=dict)  # 最近度量更新时间
    metric_validity_period: float = 1.0  # 度量有效期(秒)

    def __post_init__(self):
        """初始化字典属性"""
        if self.links is None:
            self.links = {}
        if self.traffic_metrics is None:
            self.traffic_metrics = {}
        if self.last_metrics_update is None:
            self.last_metrics_update = {}

    def add_link(self, direction: str, target_sat, capacity: float = 25.0, queue_size: int = 200):
        """
        添加星间链路

        Args:
            direction: 链路方向('north', 'south', 'east', 'west')
            target_sat: 目标卫星对象
            capacity: 链路容量(Mbps)
            queue_size: 队列大小
        """
        from models.link import Link  # 避免循环导入
        self.links[direction] = Link(self.grid_pos, target_sat.grid_pos, capacity, queue_size)

    def update_traffic_metric(self, direction: str, metric: float):
        """
        更新特定方向的流量度量

        Args:
            direction: 链路方向
            metric: 流量度量值
        """
        self.traffic_metrics[direction] = metric
        self.last_metrics_update[direction] = time.time()

    def get_traffic_metric(self, direction: str) -> float:
        """
        获取特定方向的流量度量

        如果度量过期，返回0

        Args:
            direction: 链路方向

        Returns:
            float: 流量度量值
        """
        if direction not in self.traffic_metrics:
            return 0.0

        current_time = time.time()
        last_update = self.last_metrics_update.get(direction, 0)

        # 检查度量是否有效
        if current_time - last_update > self.metric_validity_period:
            return 0.0

        return self.traffic_metrics[direction]

    def get_queue_lengths(self) -> Dict[str, int]:
        """
        获取所有方向的队列长度

        Returns:
            dict: {方向: 队列长度}
        """
        return {direction: len(link.queue) for direction, link in self.links.items()}

    def get_all_traffic_metrics(self) -> Dict[str, float]:
        """
        获取所有有效的流量度量

        Returns:
            dict: {方向: 流量度量}
        """
        current_time = time.time()
        valid_metrics = {}

        for direction, metric in self.traffic_metrics.items():
            last_update = self.last_metrics_update.get(direction, 0)
            if current_time - last_update <= self.metric_validity_period:
                valid_metrics[direction] = metric

        return valid_metrics

    def calculate_outgoing_traffic_metric(self, receiving_direction: str = None) -> float:
        """
        计算当前节点的总体输出流量度量，用于发送给邻居

        基于公式: mnode = ∑(traffic_metric_i) / (num_directions - 1)

        Args:
            receiving_direction: 接收方向，计算时排除

        Returns:
            float: 流量度量
        """
        queue_lengths = {}
        total_length = 0
        count = 0

        for direction, link in self.links.items():
            if direction != receiving_direction:
                queue_length = len(link.queue)
                queue_lengths[direction] = queue_length
                total_length += queue_length
                count += 1

        # 防止除以零
        if count == 0:
            return 0.0

        return total_length / count

    def get_geographical_position(self) -> Tuple[float, float]:
        """
        获取卫星的地理位置(纬度和经度)

        这是一个简化计算，实际应用需要使用精确的轨道模型

        Returns:
            tuple: (纬度, 经度)
        """
        i, j = self.grid_pos
        num_planes = 6  # 假设总共6个轨道面
        sats_per_plane = 11  # 假设每个轨道面11颗卫星

        # 计算经度 (0-360度)
        lon = (i * 360.0 / num_planes) % 360

        # 计算纬度 (-90到90度)
        lat = 90.0 - j * (180.0 / sats_per_plane)

        return lat, lon

    def is_in_polar_region(self, polar_threshold: float = 75.0) -> bool:
        """
        判断卫星是否在极地区域

        Args:
            polar_threshold: 极地纬度阈值

        Returns:
            bool: 是否在极地区域
        """
        lat, _ = self.get_geographical_position()
        return abs(lat) >= polar_threshold

    def get_neighbor_ids(self) -> Dict[str, Tuple[int, int]]:
        """
        获取所有邻居卫星的ID

        Returns:
            dict: {方向: 邻居ID}
        """
        return {direction: link.target_id for direction, link in self.links.items()}