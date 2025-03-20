# utils/metrics.py

from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np
import time
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)


class CongestionPhase:
    """拥塞阶段定义"""
    PRE_CONGESTION = "pre_congestion"  # 拥塞发生前
    DURING_CONGESTION = "during_congestion"  # 拥塞期间
    POST_CONTROL = "post_control"  # 拥塞控制后


class PerformanceMetrics:
    """
    性能指标计算

    为分布式概率拥塞控制算法提供性能评估
    """

    def __init__(self, config):
        """
        初始化性能指标收集器

        Args:
            config: 系统配置
        """
        self.config = config
        self.start_time = time.time()

        # 按周期和链路存储的测量数据
        self.metrics_by_cycle = {}
        for cycle in range(4):
            self.metrics_by_cycle[cycle] = {
                'queue_loads': {},  # {link_id: {phase: [values]}}
                'delays': [],  # 端到端时延
                'packet_stats': {},  # {link_id: {total, success, loss}}
                'probabilities': []  # 主次路径选择概率
            }

        # 控制开销度量
        self.data_message_size = 0
        self.control_message_size = 0

        # 当前周期状态
        self.current_phase = CongestionPhase.PRE_CONGESTION
        self.last_phase_change = time.time()

    def get_current_cycle(self) -> int:
        """
        获取当前周期

        Returns:
            int: 当前周期(0-3)
        """
        current_time = time.time() - self.start_time
        return min(3, int(current_time / self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL']))

    def _get_congestion_phase(self) -> str:
        """
        确定当前拥塞阶段

        Returns:
            str: 拥塞阶段
        """
        current_time = time.time() - self.start_time
        cycle_time = current_time % self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL']

        if cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION']:
            return CongestionPhase.DURING_CONGESTION
        elif cycle_time < self.config['CONGESTION_SCENARIO']['CONGESTION_DURATION'] + 15:
            return CongestionPhase.POST_CONTROL
        else:
            return CongestionPhase.PRE_CONGESTION

    def update_phase(self):
        """更新当前拥塞阶段"""
        new_phase = self._get_congestion_phase()
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.last_phase_change = time.time()

    def record_queue_load(self, link_id: str, queue_length: int, max_queue: int):
        """
        记录队列负载

        Args:
            link_id: 链路ID
            queue_length: 队列长度
            max_queue: 最大队列长度
        """
        cycle = self.get_current_cycle()
        phase = self.current_phase

        # 初始化数据结构
        if link_id not in self.metrics_by_cycle[cycle]['queue_loads']:
            self.metrics_by_cycle[cycle]['queue_loads'][link_id] = {
                CongestionPhase.PRE_CONGESTION: [],
                CongestionPhase.DURING_CONGESTION: [],
                CongestionPhase.POST_CONTROL: []
            }

        # 记录队列负载率
        queue_load = queue_length / max_queue
        self.metrics_by_cycle[cycle]['queue_loads'][link_id][phase].append(queue_load)

    def record_packet_stats(self, link_id: str, success: bool, delay: Optional[float] = None):
        """
        记录数据包统计

        Args:
            link_id: 链路ID
            success: 传输是否成功
            delay: 端到端时延(秒)
        """
        cycle = self.get_current_cycle()

        # 初始化数据结构
        if link_id not in self.metrics_by_cycle[cycle]['packet_stats']:
            self.metrics_by_cycle[cycle]['packet_stats'][link_id] = {
                'total': 0,
                'success': 0,
                'loss': 0,
                'delays': []
            }

        # 更新统计
        stats = self.metrics_by_cycle[cycle]['packet_stats'][link_id]
        stats['total'] += 1

        if success:
            stats['success'] += 1
            if delay is not None:
                stats['delays'].append(delay)
                self.metrics_by_cycle[cycle]['delays'].append(delay)
        else:
            stats['loss'] += 1

    def record_routing_probability(self, primary_prob: float):
        """
        记录路由概率

        Args:
            primary_prob: 选择主路径的概率
        """
        cycle = self.get_current_cycle()
        self.metrics_by_cycle[cycle]['probabilities'].append(primary_prob)

    def record_control_message(self, size: int):
        """
        记录控制消息开销

        Args:
            size: 消息大小(bytes)
        """
        self.control_message_size += size

    def record_data_message(self, size: int):
        """
        记录数据消息开销

        Args:
            size: 消息大小(bytes)
        """
        self.data_message_size += size

    def get_average_queue_load(self, link_id: str, phase: str, cycle: int) -> float:
        """
        计算平均队列负载

        Args:
            link_id: 链路ID
            phase: 拥塞阶段
            cycle: 周期

        Returns:
            float: 平均队列负载率(百分比)
        """
        if cycle in self.metrics_by_cycle and link_id in self.metrics_by_cycle[cycle]['queue_loads']:
            values = self.metrics_by_cycle[cycle]['queue_loads'][link_id].get(phase, [])
            if values:
                return sum(values) / len(values) * 100
        return 0.0

    def get_packet_loss_rate(self, link_id: str, cycle: int) -> float:
        """
        计算丢包率

        Args:
            link_id: 链路ID
            cycle: 周期

        Returns:
            float: 丢包率(百分比)
        """
        if cycle in self.metrics_by_cycle and link_id in self.metrics_by_cycle[cycle]['packet_stats']:
            stats = self.metrics_by_cycle[cycle]['packet_stats'][link_id]
            if stats['total'] > 0:
                return (stats['loss'] / stats['total']) * 100
        return 0.0

    def get_average_delay(self, cycle: int) -> float:
        """
        计算平均端到端时延

        Args:
            cycle: 周期

        Returns:
            float: 平均时延(毫秒)
        """
        if cycle in self.metrics_by_cycle and self.metrics_by_cycle[cycle]['delays']:
            delays = self.metrics_by_cycle[cycle]['delays']
            return (sum(delays) / len(delays)) * 1000  # 转换为毫秒
        return 0.0

    def get_average_probability(self, cycle: int) -> float:
        """
        计算平均路由概率

        Args:
            cycle: 周期

        Returns:
            float: 平均选择主路径的概率
        """
        if cycle in self.metrics_by_cycle and self.metrics_by_cycle[cycle]['probabilities']:
            probs = self.metrics_by_cycle[cycle]['probabilities']
            return sum(probs) / len(probs)
        return 0.5  # 默认0.5

    def get_control_overhead_ratio(self) -> float:
        """
        计算控制开销比率

        Returns:
            float: 控制开销占总流量的比率
        """
        total_size = self.control_message_size + self.data_message_size
        if total_size > 0:
            return (self.control_message_size / total_size) * 100
        return 0.0

    def get_improvement_ratio(self, link_id: str, cycle: int) -> float:
        """
        计算拥塞改善率

        Args:
            link_id: 链路ID
            cycle: 周期

        Returns:
            float: 改善率(百分比)
        """
        during_load = self.get_average_queue_load(link_id, CongestionPhase.DURING_CONGESTION, cycle)
        post_load = self.get_average_queue_load(link_id, CongestionPhase.POST_CONTROL, cycle)

        if during_load > 0:
            return ((during_load - post_load) / during_load) * 100
        return 0.0

    def generate_performance_report(self) -> str:
        """
        生成性能报告

        Returns:
            str: 报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"dra_performance_{timestamp}.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 分布式概率拥塞控制性能评估报告 ===\n\n")

            # 1. 拥塞链路性能分析
            f.write("1. 拥塞链路性能分析:\n")

            # 获取监控的链路ID
            if self.config['CONGESTION_SCENARIO']['TYPE'] == 'single':
                conf = self.config['CONGESTION_SCENARIO']['SINGLE_LINK']
                link_id = f"S({conf['source_plane']},{conf['source_index']})-{conf['direction']}"
                monitored_links = [link_id]
            else:
                monitored_links = []
                for link_conf in self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']:
                    link_id = f"S({link_conf['source_plane']},{link_conf['source_index']})-{link_conf['direction']}"
                    monitored_links.append(link_id)

            # 打印每个监控链路的数据
            for link_id in monitored_links:
                f.write(f"\n链路 {link_id} 的性能指标:\n")

                for cycle in range(4):
                    cycle_start = cycle * self.config['CONGESTION_SCENARIO']['CONGESTION_INTERVAL']
                    f.write(f"\n第{cycle + 1}次拥塞周期 (开始时间: {cycle_start}s):\n")

                    # 各阶段队列负载
                    pre_load = self.get_average_queue_load(link_id, CongestionPhase.PRE_CONGESTION, cycle)
                    during_load = self.get_average_queue_load(link_id, CongestionPhase.DURING_CONGESTION, cycle)
                    post_load = self.get_average_queue_load(link_id, CongestionPhase.POST_CONTROL, cycle)

                    f.write(f"* pre_congestion阶段 队列负载率: {pre_load:.2f}%\n")
                    f.write(f"* during_congestion阶段 队列负载率: {during_load:.2f}%\n")
                    f.write(f"* post_control阶段 队列负载率: {post_load:.2f}%\n")

                    # 改善率和丢包率
                    improvement = self.get_improvement_ratio(link_id, cycle)
                    loss_rate = self.get_packet_loss_rate(link_id, cycle)

                    f.write(f"* 拥塞控制改善率: {improvement:.2f}%\n")
                    f.write(f"* 丢包率: {loss_rate:.2f}%\n")

            # 2. 路由概率分析
            f.write("\n2. 路由概率分析:\n")
            for cycle in range(4):
                avg_prob = self.get_average_probability(cycle)
                f.write(f"第{cycle + 1}个周期平均主路径选择概率: {avg_prob:.4f}\n")

            # 3. 端到端时延分析
            f.write("\n3. 端到端时延分析:\n")
            for cycle in range(4):
                avg_delay = self.get_average_delay(cycle)
                f.write(f"第{cycle + 1}个周期平均端到端时延: {avg_delay:.2f}ms\n")

            # 4. 总体性能分析
            f.write("\n4. 总体性能分析:\n")

            # 计算各周期的总体改善率
            total_improvements = []
            for cycle in range(4):
                cycle_improvements = []
                for link_id in monitored_links:
                    improvement = self.get_improvement_ratio(link_id, cycle)
                    cycle_improvements.append(improvement)

                if cycle_improvements:
                    avg_improvement = sum(cycle_improvements) / len(cycle_improvements)
                    total_improvements.append(avg_improvement)
                    f.write(f"第{cycle + 1}个周期平均改善率: {avg_improvement:.2f}%\n")

            # 计算总体平均改善率
            if total_improvements:
                avg_improvement = sum(total_improvements) / len(total_improvements)
                std_improvement = np.std(total_improvements) if len(total_improvements) > 1 else 0.0
                f.write(f"\n总体平均改善率: {avg_improvement:.2f}%\n")
                f.write(f"改善率标准差: {std_improvement:.2f}%\n")

            # 控制开销分析
            overhead_ratio = self.get_control_overhead_ratio()
            f.write(f"控制开销比率: {overhead_ratio:.2f}%\n")

        # 生成性能图表
        self.generate_performance_plots(timestamp, monitored_links)

        logger.info(f"性能报告已生成: {report_path}")
        return report_path

    def generate_performance_plots(self, timestamp: str, monitored_links: List[str]):
        """
        生成性能分析图表

        Args:
            timestamp: 时间戳
            monitored_links: 监控的链路列表
        """
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # 绘制队列负载率变化图
        self._generate_queue_load_plot(timestamp, monitored_links)

        # 绘制端到端时延变化图
        self._generate_delay_plot(timestamp)

        # 绘制丢包率变化图
        self._generate_loss_rate_plot(timestamp, monitored_links)

        # 绘制路由概率变化图
        self._generate_probability_plot(timestamp)

    def _generate_queue_load_plot(self, timestamp: str, monitored_links: List[str]):
        """
        生成队列负载率图表

        Args:
            timestamp: 时间戳
            monitored_links: 监控的链路列表
        """
        # 为简化，只取第一个监控链路
        if not monitored_links:
            return

        link_id = monitored_links[0]

        plt.figure(figsize=self.config['VISUALIZATION']['PLOT_FIGSIZE'])

        # 准备数据
        cycles = range(1, 5)  # 1-4周期
        pre_loads = []
        during_loads = []
        post_loads = []

        for cycle in range(4):
            pre_loads.append(self.get_average_queue_load(link_id, CongestionPhase.PRE_CONGESTION, cycle))
            during_loads.append(self.get_average_queue_load(link_id, CongestionPhase.DURING_CONGESTION, cycle))
            post_loads.append(self.get_average_queue_load(link_id, CongestionPhase.POST_CONTROL, cycle))

        # 绘制柱状图
        width = 0.25
        x = np.arange(len(cycles))

        plt.bar(x - width, pre_loads, width, label='拥塞前', color='#3274A1')
        plt.bar(x, during_loads, width, label='拥塞中', color='#E1812C')
        plt.bar(x + width, post_loads, width, label='控制后', color='#3A923A')

        plt.xlabel('周期')
        plt.ylabel('队列负载率 (%)')
        plt.title(f'链路 {link_id} 队列负载率变化')
        plt.xticks(x, [f'周期 {i}' for i in cycles])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图表
        plt.savefig(f"{plots_dir}/queue_load_{timestamp}.png", dpi=self.config['VISUALIZATION']['PLOT_DPI'])
        plt.close()

    def _generate_delay_plot(self, timestamp: str):
        """
        生成端到端时延图表

        Args:
            timestamp: 时间戳
        """
        plt.figure(figsize=self.config['VISUALIZATION']['PLOT_FIGSIZE'])

        # 准备数据
        cycles = range(1, 5)  # 1-4周期
        delays = []

        for cycle in range(4):
            delays.append(self.get_average_delay(cycle))

        # 绘制折线图
        plt.plot(cycles, delays, 'o-', linewidth=2, markersize=8)

        plt.xlabel('周期')
        plt.ylabel('平均端到端时延 (ms)')
        plt.title('端到端时延变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(cycles)

        # 自定义y轴范围使图表更易读
        plt.ylim(bottom=0)

        # 保存图表
        plt.savefig(f"{plots_dir}/delay_{timestamp}.png", dpi=self.config['VISUALIZATION']['PLOT_DPI'])
        plt.close()

    def _generate_loss_rate_plot(self, timestamp: str, monitored_links: List[str]):
        """
        生成丢包率图表

        Args:
            timestamp: 时间戳
            monitored_links: 监控的链路列表
        """
        if not monitored_links:
            return

        link_id = monitored_links[0]

        plt.figure(figsize=self.config['VISUALIZATION']['PLOT_FIGSIZE'])

        # 准备数据
        cycles = range(1, 5)  # 1-4周期
        loss_rates = []

        for cycle in range(4):
            loss_rates.append(self.get_packet_loss_rate(link_id, cycle))

        # 绘制折线图
        plt.plot(cycles, loss_rates, 'o-', linewidth=2, markersize=8, color='#E1812C')

        plt.xlabel('周期')
        plt.ylabel('丢包率 (%)')
        plt.title(f'链路 {link_id} 丢包率变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(cycles)

        # 自定义y轴范围使图表更易读
        plt.ylim(bottom=0)

        # 保存图表
        plt.savefig(f"{plots_dir}/loss_rate_{timestamp}.png", dpi=self.config['VISUALIZATION']['PLOT_DPI'])
        plt.close()

    def _generate_probability_plot(self, timestamp: str):
        """
        生成路由概率图表

        Args:
            timestamp: 时间戳
        """
        plt.figure(figsize=self.config['VISUALIZATION']['PLOT_FIGSIZE'])

        # 准备数据
        cycles = range(1, 5)  # 1-4周期
        probabilities = []

        for cycle in range(4):
            probabilities.append(self.get_average_probability(cycle))

        # 绘制折线图
        plt.plot(cycles, probabilities, 'o-', linewidth=2, markersize=8, color='#3274A1')

        plt.xlabel('周期')
        plt.ylabel('主路径选择概率')
        plt.title('路由概率变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(cycles)

        # 设置y轴范围
        plt.ylim(0, 1)

        # 保存图表
        plt.savefig(f"{plots_dir}/probability_{timestamp}.png", dpi=self.config['VISUALIZATION']['PLOT_DPI'])
        plt.close()

    def simulate_realistic_measurements(self):
        """
        模拟真实测量数据

        用于演示和测试，生成合理的性能数据
        """
        # 模拟链路ID
        link_id = "S(2,3)-east"

        # 每个周期的负载基线
        base_loads = {
            CongestionPhase.PRE_CONGESTION: 0.35,
            CongestionPhase.DURING_CONGESTION: 0.85,
            CongestionPhase.POST_CONTROL: [0.65, 0.55, 0.45, 0.35]  # 每周期递减
        }

        # 每个周期的丢包率基线
        base_loss_rates = [0.15, 0.08, 0.05, 0.02]  # 每周期递减

        # 每个周期的时延基线(毫秒)
        base_delays = [60, 55, 50, 45]  # 每周期递减

        # 填充模拟数据
        for cycle in range(4):
            # 队列负载
            for phase in [CongestionPhase.PRE_CONGESTION, CongestionPhase.DURING_CONGESTION]:
                base = base_loads[phase]
                for _ in range(20):  # 每阶段20个样本
                    load = base + np.random.uniform(-0.05, 0.05)
                    if link_id not in self.metrics_by_cycle[cycle]['queue_loads']:
                        self.metrics_by_cycle[cycle]['queue_loads'][link_id] = {
                            CongestionPhase.PRE_CONGESTION: [],
                            CongestionPhase.DURING_CONGESTION: [],
                            CongestionPhase.POST_CONTROL: []
                        }
                    self.metrics_by_cycle[cycle]['queue_loads'][link_id][phase].append(load)

            # 控制后负载(每周期不同)
            post_base = base_loads[CongestionPhase.POST_CONTROL][cycle]
            for _ in range(20):
                load = post_base + np.random.uniform(-0.05, 0.05)
                if link_id not in self.metrics_by_cycle[cycle]['queue_loads']:
                    self.metrics_by_cycle[cycle]['queue_loads'][link_id] = {
                        CongestionPhase.PRE_CONGESTION: [],
                        CongestionPhase.DURING_CONGESTION: [],
                        CongestionPhase.POST_CONTROL: []
                    }
                self.metrics_by_cycle[cycle]['queue_loads'][link_id][CongestionPhase.POST_CONTROL].append(load)

            # 丢包统计
            loss_rate = base_loss_rates[cycle]
            total_packets = 1000
            loss_packets = int(total_packets * loss_rate)

            self.metrics_by_cycle[cycle]['packet_stats'][link_id] = {
                'total': total_packets,
                'success': total_packets - loss_packets,
                'loss': loss_packets,
                'delays': []
            }

            # 时延
            delay_ms = base_delays[cycle]
            for _ in range(100):  # 100个样本
                delay = (delay_ms + np.random.uniform(-5, 5)) / 1000  # 转换为秒
                self.metrics_by_cycle[cycle]['delays'].append(delay)
                self.metrics_by_cycle[cycle]['packet_stats'][link_id]['delays'].append(delay)

            # 路由概率
            if cycle == 0:
                base_prob = 0.5
            else:
                base_prob = 0.5 + cycle * 0.1  # 每周期递增

            for _ in range(50):  # 50个样本
                prob = base_prob + np.random.uniform(-0.05, 0.05)
                prob = np.clip(prob, 0, 1)  # 确保在[0,1]范围内
                self.metrics_by_cycle[cycle]['probabilities'].append(prob)

        # 控制开销
        self.data_message_size = 1024 * 1024 * 100  # 100MB数据
        self.control_message_size = 1024 * 1024 * 5  # 5MB控制数据