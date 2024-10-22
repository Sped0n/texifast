import logging
import re


class _CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS: dict[int, str] = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("texifast")
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

ch.setFormatter(_CustomFormatter())
logger.addHandler(ch)


def set_log_level(level: int) -> None:
    """Set the log level of the logger.

    Args:
        level (int): The log level to set.

    """
    logger.setLevel(level)


def refine_math_block(text: str) -> str:
    """Refine the math block in the text.

    Args:
        text (str): The text to refine.

    Returns:
        str: The refined text.
    """
    pattern = r"(\$\$.*?\$\$)"
    parts: list[str] = re.split(pattern, text, flags=re.DOTALL)
    last_block_is_math_block: bool = False
    for i in range(len(parts)):
        if re.match(pattern, parts[i]):
            parts[i] = (
                f"$$\n{parts[i][2:-2]}\n$$\n\n"
                if last_block_is_math_block
                else f"\n\n$$\n{parts[i][2:-2]}\n$$\n\n"
            )
            last_block_is_math_block = True
        else:
            parts[i] = parts[i].strip()
            last_block_is_math_block = False
    result: str = "".join(parts).strip()
    return result
