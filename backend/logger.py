import logging
import logging.config
from pathlib import Path

import yaml

_logger_setup = True


def setup_logging(level: str = "INFO") -> bool:
    """Setup logging configuration."""
    global _logger_setup

    path = Path(__file__).parent / "logging.yaml"
    if not path.exists():
        msg = f"logging.yaml not found: {path}"
        raise FileNotFoundError(msg)

    config = yaml.safe_load(path.read_text())
    for _config in config["handlers"].values():
        _config["level"] = level

    logging.config.dictConfig(config=config)
    logger = logging.getLogger(__name__)
    logger.log(
        level=getattr(logging, level.upper()),
        msg=f"Logging configured with level: {level}",
    )
    _logger_setup = True
    return _logger_setup
