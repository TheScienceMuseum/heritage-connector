import logging
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
