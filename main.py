# main.py
import numpy as np
import time
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set

from core.ospf_router import OSPFRouter
from core.lsa_manager import LSAManager, LinkStateAdvertisement
from core.congestion_detector import CongestionDetector
from core.packet import DataPacket, LSAPacket, TrafficGenerator
from models.satellite import Satellite
from utils.metrics import PerformanceMetrics
from utils.config import SYSTEM_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OSPFCongestionControl:
    """基于OSPF的拥塞控制系统"""

    def __init__(self):
        """初始化系统"""
        self.config = SYSTEM_CONFIG
        self.simulation_start_time = None

        # 初始化组件
        self.ospf_router = OSPFRouter(
            self.config['NUM_ORBIT_PLANES'],
            self.config['SATS_PER_PLANE'],
            self.config['LINK_COST_ALPHA']
        )
        self.lsa_manager = LSAManager(
            self.config['CACHE_UPDATE_INTERVAL']
        )
        self.detector = CongestionDetector(
            self.config['WARNING_THRESHOLD'],
            self.config['CONGESTION_THRESHOLD'],
            self.config['CACHE_UPDATE_INTERVAL']
        )
        self.traffic_generator = TrafficGenerator(
            self.config['LINK_CAPACITY']
        )
        self.metrics = PerformanceMetrics()

        # 初始化星座
        self.satellites = self._initialize_constellation()
        self._setup_links()
        self._identify_ground_stations()

        # 设置OSPF路由器的卫星引用
        self.ospf_router._set_satellites(self.satellites)

    def _initialize_constellation(self) -> Dict[Tuple[int, int], Satellite]:
        """
        初始化卫星星座

        Returns:
            Dict[Tuple[int, int], Satellite]: 卫星字典 {卫星ID: 卫星对象}
        """
        satellites = {}
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                grid_pos = (i, j)
                satellites[grid_pos] = Satellite(grid_pos=grid_pos)
        return satellites

    def _setup_links(self):
        """设置卫星间链路"""
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                current = self.satellites[(i, j)]

                # 建立南北向链路
                next_j = (j + 1) % self.config['SATS_PER_PLANE']
                prev_j = (j - 1) % self.config['SATS_PER_PLANE']
                current.add_link('south', self.satellites[(i, next_j)])
                current.add_link('north', self.satellites[(i, prev_j)])

                # 建立东西向链路
                if i < self.config['NUM_ORBIT_PLANES'] - 1:
                    current.add_link('east', self.satellites[(i + 1, j)])
                if i > 0:
                    current.add_link('west', self.satellites[(i - 1, j)])

    def _identify_ground_stations(self):
        """标识地面站连接点"""
        for station in self.config['CONGESTION_SCENARIO']['GROUND_STATIONS']:
            sat_id = (station['plane'], station['index'])
            if sat_id in self.satellites:
                self.satellites[sat_id].is_ground_station = True

    def _get_hotspot_satellites(self) -> List[Tuple[int, int]]:
        """
        获取热点区域卫星ID列表

        Returns:
            List[Tuple[int, int]]: 热点区域卫星ID列表
        """
        hotspots = []
        for hotspot in self.config['CONGESTION_SCENARIO']['HOTSPOTS']:
            sat_id = (hotspot['plane'], hotspot['index'])
            if sat_id in self.satellites:
                hotspots.append(sat_id)
        return hotspots

    def _get_ground_station_satellites(self) -> List[Tuple[int, int]]:
        """
        获取地面站连接卫星ID列表

        Returns:
            List[Tuple[int, int]]: 地面站连接卫星ID列表
        """
        ground_stations = []
        for satellite in self.satellites.values():
            if satellite.is_ground_station:
                ground_stations.append(satellite.grid_pos)
        return ground_stations

    def handle_packet(self, packet: DataPacket, current_sat: Satellite) -> bool:
        """
        处理数据包

        Args:
            packet: 数据包
            current_sat: 当前卫星

        Returns:
            bool: 如果处理成功返回True，否则返回False
        """
        try:
            # 如果到达目的地，成功处理
            if current_sat.grid_pos == packet.destination:
                return True

            # 使用OSPF路由器获取下一跳方向
            next_direction = self.ospf_router.get_next_hop(current_sat, packet.destination)

            if not next_direction or next_direction not in current_sat.links:
                # 找不到下一跳或链路不存在
                return False

            # 获取链路并入队
            link = current_sat.links[next_direction]
            success = link.enqueue(packet)

            if success:
                # 记录指标
                link_id = f"S{current_sat.grid_pos[0]}-{current_sat.grid_pos[1]}-{next_direction}"
                self.metrics.record_packet_metrics(packet, link_id, True)
            else:
                # 数据包丢失
                link_id = f"S{current_sat.grid_pos[0]}-{current_sat.grid_pos[1]}-{next_direction}"
                self.metrics.record_packet_metrics(packet, link_id, False)

            return success
        except Exception as e:
            logger.error(f"Error handling packet: {str(e)}")
            return False

    def update_link_costs(self):
        """更新所有链路成本"""
        for sat_id, satellite in self.satellites.items():
            for direction, link in satellite.links.items():
                # 检查链路状态
                link_state = self.detector.check_link_state(link)

                # 计算OSPF成本
                cost = link.calculate_ospf_cost()

                # 生成链路ID
                link_id = f"S{sat_id[0]}-{sat_id[1]}-{direction}"

                # 判断是否需要更新LSA
                if self.lsa_manager.should_update(link_id, cost) or self.detector.should_update_lsa(link):
                    # 创建新的LSA
                    lsa = self.lsa_manager.create_lsa(link_id, cost, sat_id)

                    # 更新OSPF路由器的链路状态数据库
                    self.ospf_router.update_link_state(link_id, cost)

    def update_routing_tables(self):
        """更新所有卫星的路由表"""
        self.ospf_router.update_all_routing_tables(self.satellites)

    def _simulate_normal_traffic(self):
        """模拟正常流量"""
        for sat_id, satellite in self.satellites.items():
            # 生成数据包
            packets = satellite.generate_traffic(
                self.traffic_generator,
                'normal',
                self.config['NUM_ORBIT_PLANES'],
                self.config['SATS_PER_PLANE']
            )

            # 处理数据包
            for packet in packets:
                self.handle_packet(packet, satellite)

    def _simulate_hotspot_traffic(self):
        """模拟热点区域流量"""
        hotspots = self._get_hotspot_satellites()
        ground_stations = self._get_ground_station_satellites()

        # 生成热点流量
        packets = self.traffic_generator.generate_hotspot_traffic(
            hotspots,
            ground_stations,
            'congestion'
        )

        # 处理数据包
        for packet in packets:
            source_id = packet.source
            if source_id in self.satellites:
                self.handle_packet(packet, self.satellites[source_id])

    def _process_link_queues(self):
        """处理所有链路队列"""
        for satellite in self.satellites.values():
            for link in satellite.links.values():
                # 出队并处理
                while True:
                    packet = link.dequeue()
                    if not packet:
                        break

                    # 获取目标卫星
                    target_id = link.target_id
                    if target_id in self.satellites:
                        target_sat = self.satellites[target_id]
                        self.handle_packet(packet, target_sat)

    def _collect_metrics(self):
        """收集性能指标"""
        current_time = time.time() - self.simulation_start_time
        cycle_time = current_time % self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL']

        # 确定当前阶段
        if cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION']:
            phase = 'during_congestion'
        elif cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION'] + 15:
            phase = 'post_control'
        else:
            phase = 'pre_congestion'

        # 更新监控链路的指标
        for link_conf in (
                [self.config['CONGESTION_SCENARIO']['SINGLE_LINK']]
                if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single'
                else self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']
        ):
            sat = self.satellites.get((link_conf['source_plane'], link_conf['source_index']))
            if sat and link_conf['direction'] in sat.links:
                link = sat.links[link_conf['direction']]
                link_id = f"S{link_conf['source_plane']}-{link_conf['source_index']}-{link_conf['direction']}"

                # 记录队列负载率
                self.metrics.record_queue_load(
                    link_id,
                    phase,
                    len(link.queue),
                    link.queue_size
                )

    def _generate_performance_report(self):
        """
        生成性能报告

        Returns:
            str: 报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("reports", f"performance_report_{timestamp}.txt")
        os.makedirs("reports", exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 基于OSPF的LEO卫星星座拥塞控制性能评估报告 ===\n\n")

            # 1. 拥塞链路性能分析
            f.write("1. 拥塞链路性能分析:\n")
            link_conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
            link_id = f"S{link_conf['source_plane']}-{link_conf['source_index']}-{link_conf['direction']}"
            f.write(f"\n链路 {link_id} 的性能指标:\n")

            # 从指标中获取数据
            for cycle in range(4):
                cycle_start = cycle * 60
                f.write(f"\n第{cycle + 1}次拥塞周期 (开始时间: {cycle_start}s):\n")

                # 获取周期性能总结
                summary = self.metrics.get_cycle_summary(cycle, link_id)

                f.write(f"* pre_congestion阶段 队列负载率: {summary['pre_congestion']:.2f}%\n")
                f.write(f"* during_congestion阶段 队列负载率: {summary['during_congestion']:.2f}%\n")
                f.write(f"* post_control阶段 队列负载率: {summary['post_control']:.2f}%\n")
                f.write(f"* 拥塞控制改善率: {summary['improvement']:.2f}%\n")

                # 获取丢包率
                loss_rates = self.metrics.calculate_link_loss_rate(link_id)
                f.write(f"* 丢包率: {loss_rates[cycle]:.2f}%\n")

            # 2. OSPF路由性能分析
            f.write("\n2. OSPF路由性能分析:\n")
            for cycle in range(4):
                f.write(f"第{cycle + 1}个周期:\n")

                # 获取每个周期的路由更新统计
                route_updates = self.metrics.get_routing_updates_stats(cycle)
                f.write(f"* LSA更新次数: {route_updates['lsa_updates']}\n")
                f.write(f"* 路由表更新次数: {route_updates['route_updates']}\n")
                f.write(f"* 平均链路成本变化: {route_updates['avg_cost_change']:.2f}\n")
                f.write(f"* 响应时间: {route_updates['response_time']:.2f}s\n")

            # 3. 总体改善效果
            f.write("\n3. 总体改善效果:\n")
            avg_improvement, std_improvement = self.metrics.calculate_overall_improvement()
            overhead = self.metrics.calculate_control_overhead()

            f.write(f"* 平均改善率: {avg_improvement:.2f}%\n")
            f.write(f"* 改善率标准差: {std_improvement:.2f}%\n")
            f.write(f"* 控制开销比例: {overhead:.2f}%\n")

        # 生成可视化图表
        self.generate_performance_plots(timestamp)

        logger.info(f"Performance report generated: {report_path}")
        return report_path

    def generate_performance_plots(self, timestamp: str):
        """
        生成性能分析图表

        Args:
            timestamp: 时间戳
        """
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # 获取链路信息
        link_conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
        link_id = f"S{link_conf['source_plane']}-{link_conf['source_index']}-{link_conf['direction']}"

        # 1. 队列负载率对比图
        plt.figure(figsize=(12, 6))

        # 从指标中获取数据
        pre_loads = []
        during_loads = []
        post_loads = []
        loss_rates = []

        for cycle in range(4):
            summary = self.metrics.get_cycle_summary(cycle, link_id)
            pre_loads.append(summary['pre_congestion'])
            during_loads.append(summary['during_congestion'])
            post_loads.append(summary['post_control'])

            rates = self.metrics.calculate_link_loss_rate(link_id)
            loss_rates.append(rates[cycle])

        # 创建柱状图和折线图组合
        x = np.arange(4)
        width = 0.25

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.bar(x - width, pre_loads, width, label='拥塞前', color='#3274A1')
        ax1.bar(x, during_loads, width, label='拥塞期间', color='#E1812C')
        ax1.bar(x + width, post_loads, width, label='控制后', color='#3A923A')

        ax1.set_ylabel('队列负载率 (%)')
        ax1.set_xlabel('周期')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'周期 {i + 1}' for i in x])
        ax1.set_ylim(0, 100)

        # 创建次坐标轴用于丢包率折线图
        ax2 = ax1.twinx()
        ax2.plot(x, loss_rates, 'k--o', linewidth=1.5, label='丢包率')
        ax2.set_ylabel('丢包率 (%)')
        ax2.set_ylim(0, 25)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax1.grid(True)
        ax1.set_title('基于OSPF的拥塞控制效果 - 队列负载率与丢包率')

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/queue_load_rate_{timestamp}.png")
        plt.close()

        # 2. 链路成本变化图
        plt.figure(figsize=(10, 6))

        cost_data = self.metrics.get_link_cost_data(link_id)
        cycles = range(4)

        for cycle in cycles:
            times = cost_data[cycle]['times']
            costs = cost_data[cycle]['costs']
            plt.plot(times, costs, 'o-', label=f'周期 {cycle + 1}')

        plt.xlabel('周期内时间 (s)')
        plt.ylabel('OSPF链路成本')
        plt.title('OSPF链路成本随时间变化')
        plt.legend()
        plt.grid(True)

        plt.savefig(f"{plots_dir}/link_cost_{timestamp}.png")
        plt.close()

    def run_simulation(self):
        """运行仿真"""
        logger.info("Starting simulation...")
        self.simulation_start_time = time.time()
        simulation_duration = self.config['CONGESTION_SCENARIO']['TOTAL_DURATION']

        # 初始化路由表
        self.update_routing_tables()

        try:
            while (time.time() - self.simulation_start_time < simulation_duration):
                current_time = time.time() - self.simulation_start_time

                # 每30秒打印进度
                if int(current_time) % 30 == 0:
                    progress = (current_time / simulation_duration) * 100
                    logger.info(f"Simulation progress: {progress:.1f}%")

                # 确定当前时间在周期中的位置
                cycle_time = current_time % self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL']

                # 检查是否处于拥塞高峰期
                is_congestion_period = (
                        cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION']
                )

                # 模拟正常流量
                self._simulate_normal_traffic()

                # 如果处于拥塞高峰期，增加热点流量
                if is_congestion_period:
                    self._simulate_hotspot_traffic()

                # 处理链路队列
                self._process_link_queues()

                # 更新链路成本
                self.update_link_costs()

                # 定期更新路由表（每5秒更新一次）
                if int(current_time) % 5 == 0:
                    self.update_routing_tables()

                # 收集性能指标
                self._collect_metrics()

                # 等待下一个仿真步
                time.sleep(self.config['SIMULATION_STEP'])

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self._generate_performance_report()


def main():
    """主函数"""
    try:
        ospf_system = OSPFCongestionControl()
        ospf_system.run_simulation()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise e


if __name__ == "__main__":
    main()