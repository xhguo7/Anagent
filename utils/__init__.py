"""
Logging utilities.
"""

import logging
import sys
import os

# ANSI color codes for logging
class LogColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @classmethod
    def disable_colors(cls):
        """Disable colors (useful for file logging or non-terminal output)"""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.PURPLE = ''
        cls.CYAN = cls.WHITE = cls.BOLD = cls.END = ''

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    LEVEL_COLORS = {
        'DEBUG': LogColors.CYAN,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.RED + LogColors.BOLD
    }
    
    def format(self, record):
        # Get the original formatted message
        message = super().format(record)
        
        # Add color based on log level
        color = self.LEVEL_COLORS.get(record.levelname, '')
        if color:
            # Color the level name and keep the rest readable
            parts = message.split(' - ', 3)  # Split on ' - ' up to 3 times
            if len(parts) >= 3:
                # Format: timestamp - name - level - message
                timestamp, name, level_and_message = parts[0], parts[1], ' - '.join(parts[2:])
                level_parts = level_and_message.split(' - ', 1)
                if len(level_parts) == 2:
                    level, msg = level_parts
                    message = f"{timestamp} - {name} - {color}{level}{LogColors.END} - {msg}"
            else:
                # Fallback: color the entire message
                message = f"{color}{message}{LogColors.END}"
        
        return message

def setup_colored_logger(name: str, level: int = logging.INFO, 
                        format_string: str = None, 
                        disable_colors: bool = False) -> logging.Logger:
    """Set up a logger with colored output."""
    
    # Disable colors if requested or if not running in a terminal
    if disable_colors or not sys.stdout.isatty():
        LogColors.disable_colors()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Set format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create and set the colored formatter
    formatter = ColoredFormatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    return logger

def setup_file_logger(name: str, filename: str, level: int = logging.INFO,
                     format_string: str = None) -> logging.Logger:
    """Set up a logger that writes to a file (without colors)."""
    logger = logging.getLogger(f"{name}_file")
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    
    # Set format (no colors for file)
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger

def setup_dual_logger(name: str, filename: str = None, 
                     console_level: int = logging.INFO,
                     file_level: int = logging.DEBUG) -> logging.Logger:
    """Set up a logger that outputs to both console (colored) and file (plain)."""
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level if filename else console_level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Only use colors if outputting to terminal
    if sys.stdout.isatty():
        console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler without colors (if filename provided)
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger

# Convenience function for quick setup
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Quick setup for colored console logger."""
    return setup_colored_logger(name, level)

# Example usage and testing
