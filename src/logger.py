import os
import sys
import logging
from datetime import datetime


logging_str = "%(asctime)s - file %(name)s of '%(lineno)d' no line - %(levelname)s - %(message)s"
log_dir = "logs"
LOG_FILE = f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
log_file_path = os.path.join(log_dir, LOG_FILE)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format=logging_str,
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger("src")


if __name__ == "__main__":
    logger.info("Logging is working")
