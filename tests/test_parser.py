import unittest
import os
import json
import yaml
import argparse
import torch # Added torch import
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel
from irontorch.distributed.parser import (
    parse_min_max_nodes,
    local_world_size,
    get_rdzv_endpoint,
    elastic_config,
    add_elastic_args,
    load_config,
    map_config,
    setup_config,
)
# Removed unused import: from irontorch.distributed.schema import MainConfig


class TestParserFunctions(unittest.TestCase):

    def test_parse_min_max_nodes(self):
        self.assertEqual(parse_min_max_nodes("1"), (1, 1))
        self.assertEqual(parse_min_max_nodes("1:2"), (1, 2))
        with self.assertRaises(ValueError):
            parse_min_max_nodes("1:2:3")

    def test_local_world_size(self):
        self.assertEqual(local_world_size("1"), 1)
        self.assertEqual(local_world_size("cpu"), os.cpu_count())
        if torch.cuda.is_available():
            self.assertEqual(local_world_size("gpu"), torch.cuda.device_count())
        else:
            with self.assertRaises(ValueError):
                local_world_size("gpu")

    def test_get_rdzv_endpoint(self):
        args = argparse.Namespace(
            rdzv_backend="static", rdzv_endpoint="", dist_url="127.0.0.1:29500"
        )
        self.assertEqual(get_rdzv_endpoint(args, 1), "127.0.0.1:29500")
        args.rdzv_endpoint = "127.0.0.1:29501"
        self.assertEqual(get_rdzv_endpoint(args, 1), "127.0.0.1:29501")

    def test_elastic_config(self):
        args = argparse.Namespace(
            n_node="1:2",
            n_proc="1",
            rdzv_backend="static",
            rdzv_endpoint="127.0.0.1:29500",
            rdzv_id="none",
            rdzv_conf="",
            node_rank=0,
            max_restarts=0,
            monitor_interval=5,
            start_method="spawn",
            role="default",
        )
        config = elastic_config(args)
        self.assertEqual(config.min_nodes, 1)
        self.assertEqual(config.max_nodes, 2)
        self.assertEqual(config.nproc_per_node, 1)
        self.assertEqual(config.rdzv_endpoint, "127.0.0.1:29500")

    def test_add_elastic_args(self):
        parser = argparse.ArgumentParser()
        parser = add_elastic_args(parser)
        args = parser.parse_args(
            ["--n_proc", "2", "--n_node", "1:2", "--node_rank", "0"]
        )
        self.assertEqual(args.n_proc, "2")
        self.assertEqual(args.n_node, "1:2")
        self.assertEqual(args.node_rank, 0)

    def test_load_config(self):
        json_config = {"key": "value"}
        yaml_config = {"key": "value"}
        with open("test_config.json", "w") as f:
            json.dump(json_config, f)
        with open("test_config.yaml", "w") as f:
            yaml.dump(yaml_config, f)
        self.assertEqual(load_config("test_config.json"), OmegaConf.create(json_config))
        self.assertEqual(load_config("test_config.yaml"), OmegaConf.create(yaml_config))
        os.remove("test_config.json")
        os.remove("test_config.yaml")
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_file.json")
        with self.assertRaises(ValueError):
            load_config("unsupported_file.txt")

    def test_map_config(self):
        config = DictConfig({"key": "value"})
        config_class = BaseModel
        mapped_config = map_config(config, config_class)
        self.assertEqual(mapped_config.key, "value")

    def test_setup_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str, default="test_config.yaml")
        args = ["--config_path", "test_config.yaml"]
        yaml_config = {"key": "value"}
        with open("test_config.yaml", "w") as f:
            yaml.dump(yaml_config, f)
        conf = setup_config(parser, args)
        self.assertEqual(conf.key, "value")
        os.remove("test_config.yaml")


if __name__ == "__main__":
    unittest.main()
