"""Logging utilities for training and tuning."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class TeeLogger:
    """Dual-stream logger that writes to both stdout and a file."""
    
    def __init__(self, log_path: Optional[str] = None):
        self.terminal = sys.stdout
        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[str] = None
        
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_path = str(log_path)
            self.write(f"=== Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def write(self, message: str):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        """Flush both streams."""
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            self.write(f"\n=== Log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            self.log_file.close()
            self.log_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StderrTeeLogger(TeeLogger):
    """Dual-stream logger for stderr."""
    
    def __init__(self, log_path: Optional[str] = None):
        self.terminal = sys.stderr
        self.log_file: Optional[TextIO] = None
        
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')


def setup_training_logger(log_dir: str, stage_name: str, command: str) -> TeeLogger:
    """
    Set up a logger for a training stage.
    
    Args:
        log_dir: Directory to write logs
        stage_name: Name of the training stage (e.g., 'lower', 'upper', 'finetune', 'tune_lower')
        command: Command being run ('train' or 'tune')
    
    Returns:
        TeeLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{command}_{stage_name}_{timestamp}.log"
    log_path = Path(log_dir) / log_filename
    
    return TeeLogger(str(log_path))


def redirect_output(logger: TeeLogger):
    """Redirect stdout to the logger."""
    sys.stdout = logger


def restore_output(original_stdout):
    """Restore original stdout."""
    sys.stdout = original_stdout
