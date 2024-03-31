import logging
from pathlib import Path

class Logger(logging.Logger):
    def __init__(self, log_dir, log_file):
        format_str = r"[%(asctime)s] %(message)s"
        logging.basicConfig(
            level=logging.INFO, datefmt=r"%Y/%m/%d %H:%M:%S", format=format_str
        )
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_dir_path / log_file))
        fh.setFormatter(logging.Formatter(format_str))
        self.addHandler(fh)
