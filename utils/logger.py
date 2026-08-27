import logging
import os
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with True Color RGB (your exact hex colors!)"""
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_code(r, g, b):
        """Generate ANSI True Color escape code"""
        return f'\033[38;2;{r};{g};{b}m'
    
    COLORS = {
        'DEBUG': rgb_code(*hex_to_rgb('#4FA5B7')),
        'INFO': rgb_code(*hex_to_rgb('#5FA767')),
        'WARNING': rgb_code(*hex_to_rgb('#ED8719')),
        'ERROR': rgb_code(*hex_to_rgb('#F35B57')),
        'CRITICAL': rgb_code(*hex_to_rgb('#F68181')),
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color attributes to the record
        log_color = self.COLORS.get(record.levelname, '')
        record.log_color = log_color
        record.reset = self.RESET
        return super().format(record)

# Warnings and errors only, on the console and in the log file alike
LOG_LEVEL = logging.WARNING


def setup_logger(name=None):
    """Set up a logger that shows filename and saves to timestamped log file."""
    # Create logs directory if it doesn't exist
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for log filename (only once, when first logger is created)
    if not hasattr(setup_logger, 'log_filename'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        setup_logger.log_filename = os.path.join(log_dir, f'app_{timestamp}.log')
    
    # Create or get logger
    logger = logging.getLogger(name or __name__)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(LOG_LEVEL)
    
    # Colored formatter for console (colors metadata only, not the message)
    color_formatter = ColoredFormatter(
        '%(log_color)s%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s%(reset)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Plain formatter for file (no colors)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(color_formatter)
    
    # File handler without colors
    file_handler = logging.FileHandler(setup_logger.log_filename)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger