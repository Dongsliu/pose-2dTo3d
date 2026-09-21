#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志系统
提供完整的日志记录、实时监控和错误追踪功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback
import json

class PoseTransferLogger:
    """
    姿势迁移系统日志器
    支持文件日志、控制台输出和Blender UI显示
    """
    
    def __init__(self, name='PoseTransfer', log_dir='logs'):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 日志文件路径
        self.main_log = self.log_dir / 'pose_transfer.log'
        self.error_log = self.log_dir / 'errors.log'
        self.debug_log = self.log_dir / 'debug.log'
        
        # 内存缓冲（用于Blender UI显示）
        self.recent_logs = []
        self.max_recent = 100
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 清除旧的handlers
        self.logger.handlers.clear()
        
        # 添加handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """设置日志处理器"""
        
        # 1. 主日志文件（INFO及以上）
        main_handler = logging.FileHandler(
            self.main_log,
            encoding='utf-8',
            mode='a'
        )
        main_handler.setLevel(logging.INFO)
        main_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)
        
        # 2. 错误日志文件（ERROR及以上）
        error_handler = logging.FileHandler(
            self.error_log,
            encoding='utf-8',
            mode='a'
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s\n'
            'File: %(pathname)s:%(lineno)d\n'
            'Function: %(funcName)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # 3. 调试日志文件（所有级别）
        debug_handler = logging.FileHandler(
            self.debug_log,
            encoding='utf-8',
            mode='a'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        debug_handler.setFormatter(debug_formatter)
        self.logger.addHandler(debug_handler)
        
        # 4. 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def _add_to_recent(self, level, message):
        """添加到最近日志缓冲"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'time': timestamp,
            'level': level,
            'message': message
        }
        self.recent_logs.append(log_entry)
        
        # 保持最大数量
        if len(self.recent_logs) > self.max_recent:
            self.recent_logs.pop(0)
    
    def debug(self, message, **kwargs):
        """调试信息"""
        self.logger.debug(message)
        self._add_to_recent('DEBUG', message)
        
        # 记录额外变量
        if kwargs:
            var_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.debug(f"  Variables: {var_str}")
    
    def info(self, message):
        """普通信息"""
        self.logger.info(message)
        self._add_to_recent('INFO', message)
    
    def warning(self, message):
        """警告信息"""
        self.logger.warning(message)
        self._add_to_recent('WARNING', message)
    
    def error(self, message, exc_info=None):
        """错误信息"""
        self.logger.error(message)
        self._add_to_recent('ERROR', message)
        
        # 记录异常堆栈
        if exc_info:
            error_trace = traceback.format_exception(*exc_info)
            for line in error_trace:
                self.logger.error(line.strip())
    
    def success(self, message):
        """成功信息"""
        self.logger.info(f"✓ {message}")
        self._add_to_recent('SUCCESS', message)
    
    def step(self, current, total, message):
        """步骤进度"""
        progress_msg = f"[{current}/{total}] {message}"
        self.logger.info(progress_msg)
        self._add_to_recent('PROGRESS', progress_msg)
    
    def separator(self):
        """分隔线"""
        sep = "=" * 60
        self.logger.info(sep)
    
    def get_recent_logs(self, count=20):
        """获取最近的日志（用于UI显示）"""
        return self.recent_logs[-count:]
    
    def clear_recent(self):
        """清空最近日志缓冲"""
        self.recent_logs.clear()
    
    def export_logs(self, output_path):
        """导出所有日志到单个文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("Pose Transfer System - Complete Logs\n")
                f.write(f"Exported: {datetime.now()}\n")
                f.write("=" * 60 + "\n\n")
                
                # 读取主日志
                if self.main_log.exists():
                    f.write("=== Main Log ===\n")
                    f.write(self.main_log.read_text(encoding='utf-8'))
                    f.write("\n\n")
                
                # 读取错误日志
                if self.error_log.exists():
                    f.write("=== Error Log ===\n")
                    f.write(self.error_log.read_text(encoding='utf-8'))
                    f.write("\n\n")
                
                # 读取调试日志（可能很大，限制行数）
                if self.debug_log.exists():
                    f.write("=== Debug Log (Last 500 lines) ===\n")
                    lines = self.debug_log.read_text(encoding='utf-8').split('\n')
                    f.write('\n'.join(lines[-500:]))
            
            return True
        except Exception as e:
            self.error(f"导出日志失败: {e}")
            return False
    
    def clear_logs(self):
        """清空所有日志文件"""
        try:
            for log_file in [self.main_log, self.error_log, self.debug_log]:
                if log_file.exists():
                    log_file.write_text('', encoding='utf-8')
            self.clear_recent()
            self.info("日志已清空")
            return True
        except Exception as e:
            self.error(f"清空日志失败: {e}")
            return False
    
    def log_system_info(self):
        """记录系统信息"""
        self.separator()
        self.info("System Information:")
        
        import platform
        self.info(f"  OS: {platform.system()} {platform.release()}")
        self.info(f"  Python: {sys.version.split()[0]}")
        
        try:
            import torch
            self.info(f"  PyTorch: {torch.__version__}")
            self.info(f"  CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        except:
            self.warning("  PyTorch: Not installed")
        
        self.separator()
    
    def log_operation_start(self, operation_name):
        """记录操作开始"""
        self.separator()
        self.info(f"Starting: {operation_name}")
        self.separator()
        return datetime.now()
    
    def log_operation_end(self, operation_name, start_time):
        """记录操作结束"""
        duration = (datetime.now() - start_time).total_seconds()
        self.separator()
        self.success(f"Completed: {operation_name} (Duration: {duration:.2f}s)")
        self.separator()


# 全局logger实例
_global_logger = None

def get_logger():
    """获取全局logger实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = PoseTransferLogger()
        _global_logger.log_system_info()
    return _global_logger

def reset_logger():
    """重置logger（用于测试）"""
    global _global_logger
    _global_logger = None


if __name__ == "__main__":
    # 测试日志系统
    logger = get_logger()
    
    logger.info("测试普通信息")
    logger.warning("测试警告信息")
    logger.debug("测试调试信息", x=10, y=20, name="test")
    logger.success("测试成功信息")
    logger.step(1, 5, "第一步完成")
    
    try:
        1 / 0
    except Exception as e:
        logger.error("测试错误信息", exc_info=sys.exc_info())
    
    print("\n最近日志:")
    for log in logger.get_recent_logs(10):
        print(f"[{log['time']}] [{log['level']}] {log['message']}")

