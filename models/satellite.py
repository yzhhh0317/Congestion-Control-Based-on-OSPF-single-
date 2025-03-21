# models/satellite.py
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import time


@dataclass
class Satellite:
    """卫星节点类"""

    grid_pos: Tuple[int, int]  # 网格坐标 (轨道面索引, 卫星索引)
    is_ground_station: bool = False  # 是否为地面站连接点
    links: Dict[str, 'Link'] = field(default_factory=dict)  # 四个方向的链路

    def __post_init__(self):
        self.links = {}
        self.routing_table = {}  # {目的地: 下一跳方向}
        self.lsa_sequence = 0  # LSA序列号
        self.last_lsa_time = 0  # 上次LSA发送时间

    def add_link(self, direction: str, target_sat, capacity: float = 25.0, distance: float = 1.0):
        """
        添加链路

        Args:
            direction: 链路方向 (north, south, east, west)
            target_sat: 目标卫星
            capacity: 链路容量(Mbps)
            distance: 链路距离
        """
        from models.link import Link  # 避免循环导入

        # 根据方向确定距离
        if direction in ['north', 'south']:
            calculated_distance = 1.0  # 同一轨道面距离为1
        elif direction in ['east', 'west']:
            calculated_distance = 2.0  # 不同轨道面距离为2
        else:
            calculated_distance = distance

        self.links[direction] = Link(
            source_id=self.grid_pos,
            target_id=target_sat.grid_pos,
            capacity=capacity,
            distance=calculated_distance
        )

    def update_routing_table(self, destination: Tuple[int, int], next_hop: str):
        """
        更新路由表

        Args:
            destination: 目的地坐标
            next_hop: 下一跳方向
        """
        self.routing_table[destination] = next_hop

    def get_next_hop(self, destination: Tuple[int, int]) -> Optional[str]:
        """
        获取到目的地的下一跳方向

        Args:
            destination: 目的地坐标

        Returns:
            Optional[str]: 下一跳方向，如果不存在返回None
        """
        return self.routing_table.get(destination)

    def process_lsa(self, lsa_packet):
        """
        处理链路状态通告

        Args:
            lsa_packet: LSA数据包

        Returns:
            bool: 如果需要转发返回True，否则返回False
        """
        # 避免处理自己发出的LSA
        if lsa_packet.source_id == self.grid_pos:
            return False

        # 在实际系统中，这里需要更新链路状态数据库
        # 并根据需要重新计算路由表

        # 简化处理：始终转发新接收的LSA
        return True

    def forward_lsa(self, lsa_packet):
        """
        转发链路状态通告

        Args:
            lsa_packet: LSA数据包
        """
        # 向所有邻居转发LSA
        for direction, link in self.links.items():
            # 实际转发逻辑...
            pass

    def generate_traffic(self, traffic_generator, state: str,
                         num_planes: int, sats_per_plane: int) -> List:
        """
        生成流量

        Args:
            traffic_generator: 流量生成器
            state: 当前状态
            num_planes: 轨道面数量
            sats_per_plane: 每个轨道面的卫星数量

        Returns:
            List: 生成的数据包列表
        """
        return traffic_generator.generate_packets(
            source=self.grid_pos,
            state=state,
            num_planes=num_planes,
            sats_per_plane=sats_per_plane
        )