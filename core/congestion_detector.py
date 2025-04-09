from typing import Dict, List, Tuple
import time
from dataclasses import dataclass
import logging

from models.satellite import Satellite
from models.link import Link

logger = logging.getLogger(__name__)


@dataclass
class QueueStateUpdatePacket:
    """队列状态更新数据包"""

    link_id: str  # 链路标识符
    queue_occupancy: float  # 当前队列占用率
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CongestionDetector:
    """拥塞检测器"""

    def __init__(self, warning_threshold: float = 0.5,
                 congestion_threshold: float = 0.75,
                 monitoring_interval: float = 10.0):
        """
        初始化拥塞检测器

        Args:
            warning_threshold: 预警阈值（队列占用率）
            congestion_threshold: 拥塞阈值（队列占用率）
            monitoring_interval: 监控间隔(秒)
        """
        self.warning_threshold = warning_threshold
        self.congestion_threshold = congestion_threshold
        self.monitoring_interval = monitoring_interval

        self.link_states = {}  # {link_id: {'state': 状态, 'history': [历史状态]}}
        self.last_check_times = {}  # {link_id: 上次检查时间}
        self.start_time = time.time()

        # 状态统计
        self.congestion_counts = {}  # {link_id: 拥塞次数}
        self.state_durations = {}  # {link_id: {状态: 持续时间总和}}

    def check_link_state(self, link: Link) -> str:
        """
        检查链路状态

        Args:
            link: 链路对象

        Returns:
            str: 链路状态 ('normal', 'warning', 'congestion')
        """
        # 生成链路ID
        source_id = link.source_id
        target_id = link.target_id
        link_id = f"S{source_id[0]}-{source_id[1]}-S{target_id[0]}-{target_id[1]}"

        current_time = time.time()
        occupancy = link.queue_occupancy
        state = 'normal'

        # 初始化链路状态记录
        if link_id not in self.link_states:
            self.link_states[link_id] = {
                'state': 'normal',
                'history': ['normal'] * 3,  # 初始状态历史
                'last_change_time': self.start_time
            }
            self.last_check_times[link_id] = self.start_time
            self.congestion_counts[link_id] = 0
            self.state_durations[link_id] = {
                'normal': 0,
                'warning': 0,
                'congestion': 0
            }

        # 检查是否到达监控间隔
        time_since_last_check = current_time - self.last_check_times.get(link_id, 0)
        if time_since_last_check < self.monitoring_interval:
            return self.link_states[link_id]['state']

        # 更新检查时间
        self.last_check_times[link_id] = current_time

        # 确定当前状态
        if occupancy >= self.congestion_threshold:
            state = 'congestion'
            if self.link_states[link_id]['state'] != 'congestion':
                self.congestion_counts[link_id] += 1
        elif occupancy >= self.warning_threshold:
            state = 'warning'

        # 更新状态持续时间
        prev_state = self.link_states[link_id]['state']
        duration = current_time - self.link_states[link_id].get('last_change_time', self.start_time)
        self.state_durations[link_id][prev_state] += duration

        # 更新状态历史
        if state != prev_state:
            self.link_states[link_id]['last_change_time'] = current_time

        self.link_states[link_id]['state'] = state
        self.link_states[link_id]['history'].append(state)
        if len(self.link_states[link_id]['history']) > 5:  # 保留最近5个状态
            self.link_states[link_id]['history'].pop(0)

        return state

    def should_update_lsa(self, link: Link) -> bool:
        """
        判断是否应该更新链路状态通告

        Args:
            link: 链路对象

        Returns:
            bool: 如果应该更新返回True，否则返回False
        """
        # 生成链路ID
        source_id = link.source_id
        target_id = link.target_id
        link_id = f"S{source_id[0]}-{source_id[1]}-S{target_id[0]}-{target_id[1]}"

        # 获取当前状态
        current_state = self.check_link_state(link)

        # 如果是拥塞状态，应该更新LSA
        if current_state == 'congestion':
            return True

        # 如果状态发生变化，应该更新LSA
        if link_id in self.link_states:
            history = self.link_states[link_id]['history']
            if len(history) >= 2 and history[-1] != history[-2]:
                return True

        return False

    def get_congestion_frequency(self, link_id: str) -> float:
        """
        获取链路拥塞频率

        Args:
            link_id: 链路标识符

        Returns:
            float: 拥塞频率 (拥塞次数/总检查次数)
        """
        if link_id not in self.congestion_counts:
            return 0.0

        total_checks = sum(1 for t in self.last_check_times.values() if t > self.start_time)
        if total_checks == 0:
            return 0.0

        return self.congestion_counts[link_id] / total_checks

    def get_state_distribution(self, link_id: str) -> Dict[str, float]:
        """
        获取链路状态分布

        Args:
            link_id: 链路标识符

        Returns:
            Dict[str, float]: 状态分布 {状态: 比例}
        """
        if link_id not in self.state_durations:
            return {'normal': 1.0, 'warning': 0.0, 'congestion': 0.0}

        durations = self.state_durations[link_id]
        total_duration = sum(durations.values())

        if total_duration == 0:
            return {'normal': 1.0, 'warning': 0.0, 'congestion': 0.0}

        return {
            state: duration / total_duration
            for state, duration in durations.items()
        }