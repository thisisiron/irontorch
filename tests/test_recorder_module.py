import pytest
import os
import tempfile
import logging
import sys
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from irontorch.recorder.logging import setup_logging
from irontorch.recorder.trackers import WandbLogger


# Patch sys.stderr to avoid errors
@pytest.fixture
def patch_sys_stderr():
    with patch('sys.stderr'):
        yield


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_logging_config():
    """Sample logging configuration for testing."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "filename": "test.log",
            },
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": True,
            }
        },
    }


@pytest.fixture
def reset_logging():
    """Reset logging configuration after each test."""
    yield
    logging.shutdown()
    logging.root.handlers.clear()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()


@patch("yaml.safe_load")
@patch("builtins.open", new_callable=mock_open)
@patch('sys.stderr')  # Patch sys.stderr to avoid errors
def test_setup_logging_with_config(mock_stderr, mock_file, mock_yaml_load, sample_logging_config, reset_logging):
    """Test setup_logging with a valid configuration file."""
    # Patch the is_file method to return True
    with patch('pathlib.Path.is_file', return_value=True):
        # Mock the YAML loading to return our sample config
        mock_yaml_load.return_value = sample_logging_config
        
        # Call setup_logging
        with patch("logging.config.dictConfig") as mock_dict_config:
            setup_logging(config_path="fake_config.yaml")
            
            # Check that open was called with the right path
            mock_file.assert_called_once()
            
            # Check that the config was loaded
            mock_yaml_load.assert_called_once()
            
            # Check that dictConfig was called with our config
            mock_dict_config.assert_called_once_with(sample_logging_config)


@patch('sys.stderr')  # Patch sys.stderr to avoid errors
def test_setup_logging_with_custom_log_path(mock_stderr, sample_logging_config, temp_log_dir, reset_logging):
    """Test setup_logging with a custom log file path."""
    # Create a custom log path
    log_path = temp_log_dir / "custom.log"
    
    # Patch Path.is_file to return True
    with patch('pathlib.Path.is_file', return_value=True):
        # Mock the file open
        with patch('builtins.open', mock_open()):
            # Mock the YAML loading to return our sample config
            with patch("yaml.safe_load", return_value=sample_logging_config):
                # Call setup_logging with custom log path
                with patch("logging.config.dictConfig") as mock_dict_config:
                    setup_logging(config_path="fake_config.yaml", log_file_path=log_path)
                    
                    # Check that the config was updated with our custom log path
                    called_config = mock_dict_config.call_args[0][0]
                    assert called_config["handlers"]["file"]["filename"] == str(log_path)


@patch('builtins.open')
@patch('builtins.print')
@patch('sys.stderr')  # Patch sys.stderr to avoid errors
def test_setup_logging_file_not_found(mock_stderr, mock_print, mock_open, reset_logging):
    """Test setup_logging when the config file is not found."""
    # Patch is_file to return False to simulate file not found
    with patch('pathlib.Path.is_file', return_value=False):
        # Call setup_logging
        with patch("logging.basicConfig") as mock_basic_config:
            setup_logging(config_path="nonexistent_config.yaml")
            
            # Check that basicConfig was called
            mock_basic_config.assert_called_once()
            
            # Check that a warning was printed
            mock_print.assert_called()


# WandbLogger tests with mocks
class TestWandbLogger:
    """Test suite for WandbLogger with mocked dependencies."""

    @patch("irontorch.recorder.trackers.wandb", None)
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=True)
    def test_wandb_not_installed(self, mock_is_primary):
        """Test WandbLogger when wandb module is not installed."""
        logger = WandbLogger(project="test_project")
        assert logger.wandb_instance is None

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=True)
    def test_wandb_logger_primary_process(self, mock_is_primary, mock_wandb):
        """Test WandbLogger initialization for primary process."""
        mock_run = MagicMock()
        mock_run.name = "test-run-123"
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(
            project="test_project",
            config={"learning_rate": 0.01},
            name="test_run"
        )

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            config={"learning_rate": 0.01},
            group=None,
            name="test_run",
            notes=None,
            resume=None,
            tags=None,
            id=None,
        )
        assert logger.wandb_instance == mock_run

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=False)
    def test_wandb_logger_non_primary_process(self, mock_is_primary, mock_wandb):
        """Test WandbLogger initialization for non-primary process."""
        logger = WandbLogger(project="test_project")

        mock_wandb.init.assert_not_called()
        assert logger.wandb_instance is None

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=True)
    def test_wandb_logger_log(self, mock_is_primary, mock_wandb):
        """Test WandbLogger.log method."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test_project")
        logger.log({"loss": 0.5, "accuracy": 0.9}, step=10)

        mock_run.log.assert_called_once_with(
            {"loss": 0.5, "accuracy": 0.9}, step=10
        )

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=True)
    def test_wandb_logger_finish(self, mock_is_primary, mock_wandb):
        """Test WandbLogger.finish method."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test_project")
        logger.finish()

        mock_run.finish.assert_called_once()
        assert logger.wandb_instance is None

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=True)
    def test_wandb_logger_init_exception(self, mock_is_primary, mock_wandb):
        """Test WandbLogger handles initialization exceptions."""
        mock_wandb.init.side_effect = Exception("Connection failed")

        logger = WandbLogger(project="test_project")

        assert logger.wandb_instance is None

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=True)
    def test_wandb_logger_log_exception(self, mock_is_primary, mock_wandb):
        """Test WandbLogger.log handles exceptions gracefully."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.log.side_effect = Exception("Log failed")
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test_project")
        # Should not raise exception
        logger.log({"loss": 0.5})

    @patch("irontorch.recorder.trackers.wandb")
    @patch("irontorch.recorder.trackers.dist.is_primary", return_value=False)
    def test_wandb_logger_log_non_primary(self, mock_is_primary, mock_wandb):
        """Test WandbLogger.log does nothing for non-primary process."""
        logger = WandbLogger(project="test_project")
        logger.log({"loss": 0.5})

        # wandb.init should not have been called
        mock_wandb.init.assert_not_called()


# DistributedLogger tests
class TestDistributedLogger:
    """Test suite for DistributedLogger."""

    def test_make_distributed_returns_distributed_logger(self):
        """Test that make_distributed wraps a logger in DistributedLogger."""
        from irontorch.recorder import make_distributed, DistributedLogger

        base_logger = logging.getLogger("test_logger")
        distributed_logger = make_distributed(base_logger)

        assert isinstance(distributed_logger, DistributedLogger)

    @patch("irontorch.distributed.is_primary", return_value=True)
    def test_logs_on_primary_in_non_distributed(self, mock_is_primary, caplog):
        """Test that logger outputs when on primary process."""
        from irontorch.recorder import make_distributed

        base_logger = logging.getLogger("test_primary")
        base_logger.setLevel(logging.DEBUG)
        logger = make_distributed(base_logger)

        with caplog.at_level(logging.INFO):
            logger.info("테스트 메시지")

        assert "테스트 메시지" in caplog.text

    @patch("irontorch.distributed.is_primary", return_value=False)
    def test_silent_on_non_primary(self, mock_is_primary, caplog):
        """Test that logger is silent when on non-primary process."""
        from irontorch.recorder import make_distributed

        base_logger = logging.getLogger("test_non_primary")
        base_logger.setLevel(logging.DEBUG)
        logger = make_distributed(base_logger)

        with caplog.at_level(logging.INFO):
            logger.info("이 메시지는 출력되지 않아야 함")

        assert "이 메시지는 출력되지 않아야 함" not in caplog.text

    @patch("irontorch.distributed.is_primary", return_value=False)
    def test_info_all_outputs_on_non_primary(self, mock_is_primary, caplog):
        """Test that info_all outputs on non-primary process."""
        from irontorch.recorder import make_distributed

        base_logger = logging.getLogger("test_info_all")
        base_logger.setLevel(logging.DEBUG)
        logger = make_distributed(base_logger)

        with caplog.at_level(logging.INFO):
            logger.info_all("모든 rank에서 출력")

        assert "모든 rank에서 출력" in caplog.text

    def test_access_original_logger_attributes(self):
        """Test that original logger attributes are accessible."""
        from irontorch.recorder import make_distributed

        base_logger = logging.getLogger("test_attrs")
        base_logger.setLevel(logging.WARNING)
        logger = make_distributed(base_logger)

        assert logger.name == "test_attrs"
        assert logger.level == logging.WARNING

    @patch("irontorch.distributed.is_primary", return_value=True)
    def test_all_log_levels_work(self, mock_is_primary, caplog):
        """Test that all log levels (debug, info, warning, error, critical) work."""
        from irontorch.recorder import make_distributed

        base_logger = logging.getLogger("test_levels")
        base_logger.setLevel(logging.DEBUG)
        logger = make_distributed(base_logger)

        with caplog.at_level(logging.DEBUG):
            logger.debug("debug message")
            logger.info("info message")
            logger.warning("warning message")
            logger.error("error message")
            logger.critical("critical message")

        assert "debug message" in caplog.text
        assert "info message" in caplog.text
        assert "warning message" in caplog.text
        assert "error message" in caplog.text
        assert "critical message" in caplog.text

    @patch("irontorch.distributed.is_primary", return_value=False)
    def test_all_log_levels_all_methods_work(self, mock_is_primary, caplog):
        """Test that all _all methods work on non-primary."""
        from irontorch.recorder import make_distributed

        base_logger = logging.getLogger("test_levels_all")
        base_logger.setLevel(logging.DEBUG)
        logger = make_distributed(base_logger)

        with caplog.at_level(logging.DEBUG):
            logger.debug_all("debug all")
            logger.info_all("info all")
            logger.warning_all("warning all")
            logger.error_all("error all")
            logger.critical_all("critical all")

        assert "debug all" in caplog.text
        assert "info all" in caplog.text
        assert "warning all" in caplog.text
        assert "error all" in caplog.text
        assert "critical all" in caplog.text