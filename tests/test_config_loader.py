import unittest
from unittest.mock import patch, mock_open

import yaml

from src.config_loader import load_config


class TestConfigLoader(unittest.TestCase):

    @patch(
            "builtins.open",
            new_callable=mock_open,
            read_data="model:\n  n_estimators: 100"
    )
    def test_load_valid_config(self, mock_file):
        """Test that a valid config file is loaded correctly."""
        config = load_config("config/config.yaml")
        self.assertIsInstance(config, dict)
        self.assertIn("model", config)
        self.assertEqual(config["model"]["n_estimators"], 100)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_missing_config_file(self, mock_file):
        """Test the behavior when the config file is missing."""
        with self.assertRaises(FileNotFoundError):
            load_config("config/non_existent.yaml")

    @patch("builtins.open", new_callable=mock_open, read_data=": invalid_yaml")
    def test_load_malformed_config(self, mock_file):
        """Test that a malformed YAML file raises a yaml.YAMLError."""
        with self.assertRaises(yaml.YAMLError):
            load_config("config/invalid_config.yaml")

    @patch(
            "builtins.open",
            new_callable=mock_open,
            read_data="model:\n  n_estimators: 100"
    )
    def test_load_correct_data_structure(self, mock_file):
        """Test that the returned config data is a dictionary."""
        config = load_config("config/config.yaml")
        self.assertIsInstance(config, dict)


if __name__ == "__main__":
    unittest.main()
