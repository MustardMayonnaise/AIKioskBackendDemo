import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging() -> None:
	log_dir = Path(__file__).resolve().parent.parent / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	log_path = log_dir / "app.log"

	root_logger = logging.getLogger()

	formatter = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	root_logger.setLevel(logging.INFO)

	has_console_handler = any(
		isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
		for handler in root_logger.handlers
	)
	if not has_console_handler:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)
		console_handler.setFormatter(formatter)
		root_logger.addHandler(console_handler)

	has_file_handler = any(
		isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path
		for handler in root_logger.handlers
	)
	if not has_file_handler:
		file_handler = RotatingFileHandler(
			log_path,
			maxBytes=5 * 1024 * 1024,
			backupCount=3,
			encoding="utf-8",
		)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)
