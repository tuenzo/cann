#!/usr/bin/env python3
"""
监控 CANN 实验进程的资源使用情况
================================

实时显示 CPU、内存使用情况。
"""

import psutil
import time
import argparse
import subprocess
import signal
import sys
from pathlib import Path


class ResourceMonitor:
    def __init__(self, process_name="python", interval=1.0, output_file=None):
        self.process_name = process_name
        self.interval = interval
        self.output_file = output_file
        self.running = True
        
    def get_process_info(self):
        """获取所有匹配进程的信息"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
            if proc.info['name'] and self.process_name in proc.info['name']:
                processes.append(proc.info)
        return processes
    
    def get_system_info(self):
        """获取系统资源信息"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'cpu_count_physical': cpu_count_physical,
            'memory_total_gb': mem.total / (1024**3),
            'memory_used_gb': mem.used / (1024**3),
            'memory_percent': mem.percent,
            'disk_total_gb': disk.total / (1024**3),
            'disk_used_gb': disk.used / (1024**3),
            'disk_percent': disk.percent,
        }
    
    def print_header(self):
        """打印表头"""
        header = f"\n{'='*80}"
        header += f"\n{'PID':<8} {'进程名':<15} {'CPU%':<8} {'内存%':<8} {'线程数':<8}"
        header += f"\n{'-'*80}"
        print(header)
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(header + '\n')
    
    def print_status(self, processes, system_info):
        """打印状态"""
        # 系统信息
        sys_line = f"🖥️  系统总览: CPU {system_info['cpu_percent']:.1f}% ({system_info['cpu_count']} 逻辑核心, {system_info['cpu_count_physical']} 物理核心) | "
        sys_line += f"内存 {system_info['memory_used_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB ({system_info['memory_percent']:.1f}%)"
        print(f"\r{sys_line}", end='', flush=True)
        
        # 进程信息
        if not processes:
            return
        
        print()  # 换行
        for proc in processes:
            line = f"{proc['pid']:<8} {proc['name']:<15} {proc['cpu_percent']:<8.1f} {proc['memory_percent']:<8.1f} {proc['num_threads']:<8}"
            print(line)
            
            if self.output_file:
                with open(self.output_file, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {line}\n")
    
    def monitor(self, duration=None):
        """开始监控
        
        Args:
            duration: 监控时长（秒），None 表示持续监控
        """
        import signal
        
        # 信号处理
        def signal_handler(sig, frame):
            self.running = False
            print("\n\n监控已停止")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        self.print_header()
        
        start_time = time.time()
        
        while self.running:
            if duration and (time.time() - start_time) >= duration:
                break
            
            processes = self.get_process_info()
            system_info = self.get_system_info()
            self.print_status(processes, system_info)
            
            time.sleep(self.interval)
        
        print(f"\n\n监控结束")


def main():
    parser = argparse.ArgumentParser(description='监控实验进程的资源使用情况')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='采样间隔（秒，默认 1.0）')
    parser.add_argument('--duration', type=int, default=None,
                        help='监控时长（秒，None 表示持续监控）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出日志文件路径')
    parser.add_argument('--pid', type=int, default=None,
                        help='监控指定 PID 的进程')
    args = parser.parse_args()
    
    monitor = ResourceMonitor(
        process_name="python",
        interval=args.interval,
        output_file=args.output
    )
    
    print("="*80)
    print("资源监控器")
    print("="*80)
    print(f"采样间隔: {args.interval} 秒")
    print(f"监控时长: {args.duration if args.duration else '持续监控'}")
    if args.output:
        print(f"输出日志: {args.output}")
    print("按 Ctrl+C 停止")
    print("="*80)
    
    monitor.monitor(duration=args.duration)


if __name__ == '__main__':
    main()

