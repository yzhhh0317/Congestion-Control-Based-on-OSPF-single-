# core/lsa_manager.py
from typing import Dict, Tuple, List, Set
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LinkStateAdvertisement:
    """链路状态通告(LSA)数据结构"""

    link_id: str  # 链路标识符 "S轨道面-卫星-方向"，例如 "S0-1-east"
    cost: float  # 链路成本
    source_id: Tuple[int, int]  # 源卫星ID
    sequence_number: int  # 序列号
    timestamp: float  # 时间戳

    def __post_init__(self):
        if not hasattr(self, 'timestamp'):
            self.timestamp = time.time()


class LSAManager:
    """链路状态通告管理器"""

    def __init__(self, update_interval: float = 10.0):
        """
        初始化LSA管理器

        Args:
            update_interval: 链路状态更新间隔(秒)
        """
        self.update_interval = update_interval
        self.lsa_database = {}  # {link_id: LinkStateAdvertisement}
        self.sequence_numbers = {}  # {link_id: 当前序列号}
        self.last_update_times = {}  # {link_id: 上次更新时间}
        self.start_time = time.time()

    def create_lsa(self, link_id: str, cost: float, source_id: Tuple[int, int]) -> LinkStateAdvertisement:
        """
        创建新的LSA

        Args:
            link_id: 链路标识符
            cost: 链路成本
            source_id: 源卫星ID

        Returns:
            LinkStateAdvertisement: 新的LSA
        """
        # 获取或初始化序列号
        if link_id not in self.sequence_numbers:
            self.sequence_numbers[link_id] = 0
        else:
            self.sequence_numbers[link_id] += 1

        # 创建LSA
        lsa = LinkStateAdvertisement(
            link_id=link_id,
            cost=cost,
            source_id=source_id,
            sequence_number=self.sequence_numbers[link_id],
            timestamp=time.time()
        )

        # 更新数据库
        self.lsa_database[link_id] = lsa
        self.last_update_times[link_id] = time.time()

        return lsa

    def should_update(self, link_id: str, current_cost: float) -> bool:
        """
        判断是否应该更新LSA

        Args:
            link_id: 链路标识符
            current_cost: 当前链路成本

        Returns:
            bool: 如果需要更新返回True，否则返回False
        """
        current_time = time.time()

        # 检查是否到达更新间隔
        if link_id in self.last_update_times:
            time_since_last_update = current_time - self.last_update_times[link_id]
            if time_since_last_update < self.update_interval:
                return False

        # 检查成本是否变化
        if link_id in self.lsa_database:
            old_lsa = self.lsa_database[link_id]
            if abs(old_lsa.cost - current_cost) < 0.05:  # 成本变化不大时不更新
                return False

        return True

    def process_lsa(self, lsa: LinkStateAdvertisement) -> bool:
        """
        处理接收到的LSA

        Args:
            lsa: 接收到的LSA

        Returns:
            bool: 如果是新的或更新的LSA返回True，否则返回False
        """
        link_id = lsa.link_id

        # 检查是否已有该LSA
        if link_id in self.lsa_database:
            old_lsa = self.lsa_database[link_id]

            # 如果接收到的LSA序列号较小，忽略
            if lsa.sequence_number < old_lsa.sequence_number:
                return False

            # 如果序列号相同但时间戳较旧，忽略
            if lsa.sequence_number == old_lsa.sequence_number and lsa.timestamp <= old_lsa.timestamp:
                return False

        # 更新LSA数据库
        self.lsa_database[link_id] = lsa
        return True

    def get_all_lsas(self) -> List[LinkStateAdvertisement]:
        """
        获取所有LSA

        Returns:
            List[LinkStateAdvertisement]: LSA列表
        """
        return list(self.lsa_database.values())

    def get_link_cost(self, link_id: str) -> float:
        """
        获取链路成本

        Args:
            link_id: 链路标识符

        Returns:
            float: 链路成本，如果不存在返回默认值1.0
        """
        if link_id in self.lsa_database:
            return self.lsa_database[link_id].cost
        return 1.0  # 默认成本