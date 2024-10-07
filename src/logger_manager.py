import logging


class LoggerManager:
    def __init__(self, name=__name__, log_level=logging.DEBUG):
        """Initialize the LoggerManager
        with a specific log level and format."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self._setup_console_handler()

    def _setup_console_handler(self):
        """Set up the console handler with a formatter."""
        # Check if the logger already has handlers
        # to prevent adding multiple handlers
        if not self.logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

    def get_logger(self):
        """Return the configured logger."""
        return self.logger
