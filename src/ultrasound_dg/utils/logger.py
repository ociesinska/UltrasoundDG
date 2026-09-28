import logging


def format_logger() -> None:
    """Configure consistent timestamped INFO-level logging for CLI scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
