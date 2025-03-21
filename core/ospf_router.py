# core/ospf_router.py
from typing import Dict, Tuple, List, Set
import heapq
import logging
from models.satellite import Satellite
from models.link import Link

logger = logging.getLogger(__name__)


class OSPFRouter:
    """OSPF路由器实现"""

    def __init__(self, num_planes: int = 6, sats_per_plane: int = 11,
                 link_cost_alpha: float = 0.5):
        """
        初始化OSPF路由器

        Args:
            num_planes: 轨道面数量
            sats_per_plane: 每个轨道面的卫星数量
            link_cost_alpha: 链路成本中缓存影响因子
        """
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.link_cost_alpha = link_cost_alpha
        self.routing_tables = {}  # {sat_id: {destination: next_hop}}
        self.link_state_database = {}  # LSDB: {link_id: cost}
        self.last_update_time = 0

    def update_link_state(self, link_id: str, cost: float):
        """
        更新链路状态数据库

        Args:
            link_id: 链路标识符
            cost: 链路成本
        """
        self.link_state_database[link_id] = cost

    def calculate_shortest_paths(self, source_sat: Satellite):
        """
        使用Dijkstra算法计算从源卫星到所有其他卫星的最短路径

        Args:
            source_sat: 源卫星节点
        """
        source_id = source_sat.grid_pos
        distances = {source_id: 0}  # 距离表
        previous = {}  # 前驱节点表
        next_hops = {}  # 下一跳表

        # 创建优先队列
        pq = [(0, source_id)]

        # 已处理节点集合
        processed = set()

        while pq:
            # 获取当前距离最小的节点
            current_distance, current_id = heapq.heappop(pq)

            # 如果节点已处理，跳过
            if current_id in processed:
                continue

            processed.add(current_id)

            # 获取当前节点的邻居
            current_sat = self._get_satellite_by_id(current_id)
            if not current_sat:
                continue

            # 遍历邻居节点
            for direction, link in current_sat.links.items():
                neighbor_id = link.target_id

                # 获取链路成本
                link_id = f"S{current_id[0]}-{current_id[1]}-{direction}"
                cost = self.link_state_database.get(link_id, self._calculate_base_cost(current_id, neighbor_id))

                # 计算通过当前节点到达邻居的距离
                distance = current_distance + cost

                # 如果找到更短的路径，则更新
                if neighbor_id not in distances or distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    previous[neighbor_id] = current_id

                    # 如果是源节点的直接邻居，记录实际的下一跳
                    if current_id == source_id:
                        next_hops[neighbor_id] = direction
                    # 否则，继承到达当前节点的下一跳
                    elif current_id in next_hops:
                        next_hops[neighbor_id] = next_hops[current_id]

                    # 将邻居加入优先队列
                    heapq.heappush(pq, (distance, neighbor_id))

        # 更新源节点的路由表
        routing_table = {}

        # 处理间接可达的节点
        for dest_id in distances:
            if dest_id == source_id:
                continue

            # 找出到达目的节点的第一跳
            current = dest_id
            while current in previous and previous[current] != source_id:
                current = previous[current]

            if current in previous:  # 确保路径存在
                direction = next_hops.get(current)
                if direction:
                    routing_table[dest_id] = direction

        self.routing_tables[source_id] = routing_table
        return routing_table

    def _get_satellite_by_id(self, sat_id: Tuple[int, int]) -> Satellite:
        """
        根据ID获取卫星对象，实际实现时应该从系统的卫星字典中获取

        Args:
            sat_id: 卫星ID (轨道面索引, 卫星索引)

        Returns:
            Satellite: 卫星对象
        """
        # 此方法在实际使用时需要外部注入卫星字典
        # 在这里作为一个接口方法
        pass

    def _calculate_base_cost(self, source_id: Tuple[int, int], target_id: Tuple[int, int]) -> float:
        """
        计算基础链路成本（仅考虑距离）

        Args:
            source_id: 源卫星ID
            target_id: 目标卫星ID

        Returns:
            float: 链路基础成本
        """
        # 计算轨道面和同轨卫星的距离
        plane_diff = abs(source_id[0] - target_id[0])
        if plane_diff > self.num_planes // 2:
            plane_diff = self.num_planes - plane_diff

        sat_diff = abs(source_id[1] - target_id[1])
        if sat_diff > self.sats_per_plane // 2:
            sat_diff = self.sats_per_plane - sat_diff

        # 简化模型：相邻卫星距离为1，不同轨道面的距离为2
        if plane_diff == 0:  # 同一轨道面
            return sat_diff
        elif sat_diff == 0:  # 同一纬度带
            return plane_diff * 2
        else:  # 对角线连接
            return plane_diff * 2 + sat_diff

    def get_next_hop(self, source_sat: Satellite, destination: Tuple[int, int]) -> str:
        """
        获取从源卫星到目的地的下一跳方向

        Args:
            source_sat: 源卫星
            destination: 目的地坐标

        Returns:
            str: 下一跳方向（north, south, east, west）
        """
        source_id = source_sat.grid_pos

        # 如果没有路由表或路由表过期，重新计算
        if source_id not in self.routing_tables:
            self.calculate_shortest_paths(source_sat)

        # 获取下一跳
        routing_table = self.routing_tables[source_id]

        if destination in routing_table:
            return routing_table[destination]

        # 如果找不到路由，使用默认路由策略（基于坐标计算）
        return self._calculate_default_next_hop(source_id, destination)

    def _calculate_default_next_hop(self, current_pos: Tuple[int, int],
                                    target_pos: Tuple[int, int]) -> str:
        """
        计算默认下一跳方向（当路由表中没有对应条目时使用）

        Args:
            current_pos: 当前位置
            target_pos: 目标位置

        Returns:
            str: 下一跳方向
        """
        curr_i, curr_j = current_pos
        target_i, target_j = target_pos

        # 计算轨道面差和同轨卫星编号差
        delta_i = (target_i - curr_i) % self.num_planes
        if delta_i > self.num_planes // 2:
            delta_i -= self.num_planes

        delta_j = (target_j - curr_j) % self.sats_per_plane
        if delta_j > self.sats_per_plane // 2:
            delta_j -= self.sats_per_plane

        # 根据相对位置选择方向
        if abs(delta_i) > abs(delta_j):  # 轨道面差更大
            return 'east' if delta_i > 0 else 'west'
        else:  # 同轨卫星编号差更大
            return 'south' if delta_j > 0 else 'north'

    def update_all_routing_tables(self, satellites: Dict[Tuple[int, int], Satellite]):
        """
        更新所有卫星的路由表

        Args:
            satellites: 卫星字典 {卫星ID: 卫星对象}
        """
        self._set_satellites(satellites)

        # 为每个卫星计算最短路径
        for sat_id, satellite in satellites.items():
            self.calculate_shortest_paths(satellite)

    def _set_satellites(self, satellites: Dict[Tuple[int, int], Satellite]):
        """
        设置系统中的卫星字典，用于路径计算

        Args:
            satellites: 卫星字典 {卫星ID: 卫星对象}
        """
        self.satellites = satellites

    def _get_satellite_by_id(self, sat_id: Tuple[int, int]) -> Satellite:
        """
        根据ID获取卫星对象

        Args:
            sat_id: 卫星ID

        Returns:
            Satellite: 卫星对象
        """
        return self.satellites.get(sat_id)