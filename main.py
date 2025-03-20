# main.py
import numpy as np
import time
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set

# 导入核心组件
from core.dra_router import DRARouter
from core.probabilistic_controller import ProbabilisticController
from core.congestion_detector import CongestionDetector
from core.packet import DataPacket, TrafficGenerator, TrafficMetricPacket, QueueStateUpdatePacket

# 导入模型
from models.satellite import Satellite
from models.link import Link

# 导入工具
from utils.metrics import PerformanceMetrics, CongestionPhase
from utils.config import SYSTEM_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProbabilisticCongestionControl:
    """
    分布式概率拥塞控制系统主类

    实现DRA路由算法和分布式概率拥塞控制
    """

    def __init__(self):
        """初始化系统"""
        # 加载配置
        self.config = SYSTEM_CONFIG
        self.simulation_start_time = None

        # 设置随机数种子，确保结果可重现
        np.random.seed(self.config['SIMULATION_SEED'])

        # 初始化组件
        self.dra_router = DRARouter(
            self.config['NUM_ORBIT_PLANES'],
            self.config['SATS_PER_PLANE'],
            self.config['POLAR_THRESHOLD']
        )

        self.prob_controller = ProbabilisticController(
            buffer_weight=self.config['BUFFER_WEIGHT'],
            neighbor_weight=self.config['NEIGHBOR_WEIGHT'],
            pref_probability=self.config['PREF_PROBABILITY'],
            threshold=self.config['THRESHOLD'],
            queue_size=self.config['QUEUE_SIZE']
        )

        self.congestion_detector = CongestionDetector(
            warning_threshold=self.config['WARNING_THRESHOLD'],
            congestion_threshold=self.config['CONGESTION_THRESHOLD'],
            release_duration=self.config['RELEASE_DURATION']
        )

        # 初始化性能指标收集器
        self.metrics = PerformanceMetrics(self.config)

        # 初始化卫星星座和链路
        self.satellites = self._initialize_constellation()
        self._setup_links()

        # 初始化流量生成器
        self.traffic_generator = TrafficGenerator(self.config['LINK_CAPACITY'])

        logger.info("分布式概率拥塞控制系统初始化完成")

    def _initialize_constellation(self) -> Dict[Tuple[int, int], Satellite]:
        """
        初始化卫星星座

        Returns:
            dict: 网格坐标到卫星对象的映射
        """
        logger.info(f"初始化卫星星座: {self.config['NUM_ORBIT_PLANES']}个轨道面, "
                    f"每个轨道面{self.config['SATS_PER_PLANE']}颗卫星")

        satellites = {}
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                grid_pos = (i, j)
                satellites[grid_pos] = Satellite(grid_pos=grid_pos)

                # 标记需要监控的卫星
                if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single':
                    conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
                    if (i == conf['source_plane'] and j == conf['source_index']):
                        satellites[grid_pos].is_monitored = True

        return satellites

    def _setup_links(self):
        """建立卫星间的链路"""
        logger.info("建立星间链路")

        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                current = self.satellites[(i, j)]

                # 建立南北向链路(同一轨道内)
                next_j = (j + 1) % self.config['SATS_PER_PLANE']
                prev_j = (j - 1) % self.config['SATS_PER_PLANE']

                current.add_link('south', self.satellites[(i, next_j)],
                                 self.config['LINK_CAPACITY'],
                                 self.config['QUEUE_SIZE'])

                current.add_link('north', self.satellites[(i, prev_j)],
                                 self.config['LINK_CAPACITY'],
                                 self.config['QUEUE_SIZE'])

                # 建立东西向链路(跨轨道)
                if i < self.config['NUM_ORBIT_PLANES'] - 1:
                    current.add_link('east', self.satellites[(i + 1, j)],
                                     self.config['LINK_CAPACITY'],
                                     self.config['QUEUE_SIZE'])

                if i > 0:
                    current.add_link('west', self.satellites[(i - 1, j)],
                                     self.config['LINK_CAPACITY'],
                                     self.config['QUEUE_SIZE'])

        logger.info("星间链路建立完成")

    def handle_packet(self, packet: DataPacket, current_sat: Satellite) -> bool:
        """
        处理数据包路由，实现分布式概率拥塞控制

        Args:
            packet: 数据包
            current_sat: 当前卫星节点

        Returns:
            bool: 处理是否成功
        """
        try:
            # 已到达目标节点
            if current_sat.grid_pos == packet.destination:
                return True

            # 使用DRA计算主要和次要方向
            primary_dir, secondary_dir = self.dra_router.calculate_directions(
                current_sat.grid_pos, packet.destination, current_sat)

            if not primary_dir:
                logger.warning(f"无法计算从{current_sat.grid_pos}到{packet.destination}的路由")
                return False

            # 获取队列长度信息
            queue_lengths = current_sat.get_queue_lengths()

            # 获取流量度量信息
            traffic_metrics = current_sat.get_all_traffic_metrics()

            # 使用概率控制器做出路由决策
            selected_dir = self.prob_controller.make_routing_decision(
                primary_dir, secondary_dir, queue_lengths, traffic_metrics)

            # 记录路由概率(用于性能分析)
            if primary_dir and secondary_dir:
                primary_congestion = self.prob_controller.calculate_congestion_level(
                    queue_lengths.get(primary_dir, 0),
                    traffic_metrics.get(primary_dir, 0)
                )

                secondary_congestion = self.prob_controller.calculate_congestion_level(
                    queue_lengths.get(secondary_dir, 0),
                    traffic_metrics.get(secondary_dir, 0)
                )

                prob = self.prob_controller.calculate_routing_probability(
                    primary_congestion, secondary_congestion)

                self.metrics.record_routing_probability(prob)

            # 获取选择的链路
            selected_link = current_sat.links.get(selected_dir)
            if not selected_link:
                logger.warning(f"链路不存在: {current_sat.grid_pos}-{selected_dir}")
                return False

            # 检查链路拥塞状态
            link_state = self.congestion_detector.check_link_state(selected_link)

            # 构造链路ID(用于指标收集)
            link_id = f"S{current_sat.grid_pos[0]}-{current_sat.grid_pos[1]}-{selected_dir}"

            # 更新数据包中的流量度量
            outgoing_metric = current_sat.calculate_outgoing_traffic_metric(selected_dir)
            self.prob_controller.update_packet_metric(packet, outgoing_metric)

            # 数据包入队
            success = selected_link.enqueue(packet)

            # 记录性能指标
            if success:
                delay = time.time() - packet.creation_time
                self.metrics.record_packet_stats(link_id, True, delay)
                self.metrics.record_data_message(packet.size // 8)  # 比特转字节
            else:
                self.metrics.record_packet_stats(link_id, False)

            # 更新流量度量
            if selected_link.should_update_metrics():
                # 生成并发送流量度量包
                target_sat = self.satellites.get(selected_link.target_id)
                if target_sat:
                    reverse_dir = self._get_reverse_direction(selected_dir)
                    target_sat.update_traffic_metric(reverse_dir, outgoing_metric)

                    # 记录控制消息开销
                    self.metrics.record_control_message(64)  # 假设64字节的控制消息

            return success

        except Exception as e:
            logger.error(f"处理数据包时出错: {str(e)}", exc_info=True)
            return False

    def _get_reverse_direction(self, direction: str) -> str:
        """
        获取相反方向

        Args:
            direction: 方向

        Returns:
            str: 相反方向
        """
        opposite = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        return opposite.get(direction, direction)

    def _simulate_packet_transmission(self):
        """
        模拟数据包传输

        生成流量并传输数据包
        """
        current_time = time.time() - self.simulation_start_time
        current_cycle = self.metrics.get_current_cycle()

        # 更新当前拥塞阶段
        self.metrics.update_phase()
        current_phase = self.metrics.current_phase

        # 针对单链路拥塞场景生成流量
        if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single':
            conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
            source_sat = self.satellites.get((conf['source_plane'], conf['source_index']))

            if source_sat and source_sat.is_monitored:
                # 针对拥塞链路生成特定状态的流量
                link = source_sat.links.get(conf['direction'])

                if link:
                    # 构造链路ID
                    link_id = f"S({conf['source_plane']},{conf['source_index']})-{conf['direction']}"

                    # 生成适当状态的流量
                    state = 'normal'
                    if current_phase == CongestionPhase.DURING_CONGESTION:
                        state = 'congestion'
                    elif current_phase == CongestionPhase.POST_CONTROL:
                        state = 'warning'

                    # 生成数据包
                    packets = self.traffic_generator.generate_packets(
                        source_sat.grid_pos,
                        state,
                        self.config['NUM_ORBIT_PLANES'],
                        self.config['SATS_PER_PLANE']
                    )

                    # 将数据包路由到网络中
                    for packet in packets:
                        self.handle_packet(packet, source_sat)

                    # 记录队列负载
                    self.metrics.record_queue_load(
                        link_id,
                        len(link.queue),
                        link.queue_size
                    )

        # 为所有其他卫星生成背景流量
        for pos, sat in self.satellites.items():
            # 跳过拥塞源卫星(已经处理过)
            if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single':
                conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
                if pos == (conf['source_plane'], conf['source_index']):
                    continue

            # 每10个卫星只为1个生成流量(降低负载)
            if np.random.random() < 0.1:
                # 生成背景流量(正常状态)
                packets = self.traffic_generator.generate_packets(
                    sat.grid_pos,
                    'normal',
                    self.config['NUM_ORBIT_PLANES'],
                    self.config['SATS_PER_PLANE']
                )

                # 将数据包路由到网络中
                for packet in packets:
                    self.handle_packet(packet, sat)

        # 出队并转发数据包
        self._process_queues()

    def _process_queues(self):
        """处理所有队列，出队并转发数据包"""
        for sat in self.satellites.values():
            for direction, link in sat.links.items():
                # 尝试出队一个数据包
                packet = link.dequeue()

                if packet:
                    # 获取目标卫星
                    target_sat = self.satellites.get(link.target_id)

                    if target_sat:
                        # 提取数据包中的流量度量信息
                        reverse_dir = self._get_reverse_direction(direction)
                        target_sat.update_traffic_metric(reverse_dir, packet.traffic_metric)

                        # 继续处理数据包
                        self.handle_packet(packet, target_sat)

    def run_simulation(self):
        """
        运行仿真

        根据配置运行指定时长的仿真
        """
        logger.info("开始仿真...")
        self.simulation_start_time = time.time()
        simulation_duration = self.config['CONGESTION_SCENARIO']['TOTAL_DURATION']

        try:
            last_progress_report = 0

            while (time.time() - self.simulation_start_time < simulation_duration):
                current_time = time.time() - self.simulation_start_time

                # 每30秒报告进度
                if int(current_time) // 30 > last_progress_report:
                    progress = (current_time / simulation_duration) * 100
                    logger.info(f"仿真进度: {progress:.1f}%")
                    last_progress_report = int(current_time) // 30

                # 模拟数据包传输
                self._simulate_packet_transmission()

                # 按配置的步长暂停
                time.sleep(self.config['SIMULATION_STEP'])

            logger.info("仿真完成")

        except KeyboardInterrupt:
            logger.info("用户中断仿真")
        except Exception as e:
            logger.error(f"仿真过程中出错: {str(e)}", exc_info=True)
        finally:
            # 生成性能报告
            report_path = self.metrics.generate_performance_report()
            logger.info(f"性能报告已生成: {report_path}")

    def _collect_metrics(self):
        """收集性能指标"""
        # 更新当前拥塞阶段
        self.metrics.update_phase()

        # 收集监控链路的指标
        if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single':
            conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
            sat = self.satellites.get((conf['source_plane'], conf['source_index']))

            if sat and sat.is_monitored:
                link = sat.links.get(conf['direction'])

                if link:
                    link_id = f"S({conf['source_plane']},{conf['source_index']})-{conf['direction']}"

                    # 记录队列负载
                    self.metrics.record_queue_load(
                        link_id,
                        len(link.queue),
                        link.queue_size
                    )


def main():
    """主函数"""
    try:
        # 创建并运行分布式概率拥塞控制系统
        dpc_system = ProbabilisticCongestionControl()
        dpc_system.run_simulation()
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()