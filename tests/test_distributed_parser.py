import pytest
import os
import tempfile
import yaml
import json
import argparse
from omegaconf import OmegaConf, DictConfig

from irontorch.distributed.parser import (
    parse_min_max_nodes,
    local_world_size,
    add_elastic_args,
    load_config,
    setup_config,
)


def test_parse_min_max_nodes():
    # Test single node
    min_node, max_node = parse_min_max_nodes("1")
    assert min_node == 1
    assert max_node == 1
    
    # Test min:max format
    min_node, max_node = parse_min_max_nodes("2:4")
    assert min_node == 2
    assert max_node == 4
    
    # Test invalid format
    with pytest.raises(ValueError):
        parse_min_max_nodes("1:2:3")


def test_local_world_size():
    # Test with numerical value
    assert local_world_size("2") == 2
    
    # Test with 'cpu'
    cpu_count = local_world_size("cpu")
    assert cpu_count > 0
    assert isinstance(cpu_count, int)
    
    # Test invalid value
    with pytest.raises(ValueError):
        local_world_size("invalid_value")


def test_add_elastic_args():
    parser = argparse.ArgumentParser()
    enhanced_parser = add_elastic_args(parser)
    
    # Check if all required arguments were added
    args = enhanced_parser.parse_args([])
    
    assert hasattr(args, "n_proc")
    assert hasattr(args, "n_node")
    assert hasattr(args, "node_rank")
    assert hasattr(args, "dist_url")
    assert hasattr(args, "rdzv_backend")
    assert hasattr(args, "rdzv_endpoint")
    assert hasattr(args, "rdzv_id")
    assert hasattr(args, "rdzv_conf")
    assert hasattr(args, "standalone")
    assert hasattr(args, "max_restarts")
    assert hasattr(args, "monitor_interval")
    assert hasattr(args, "start_method")
    assert hasattr(args, "role")


@pytest.fixture
def temp_config_files():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create YAML config file
        yaml_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "model": {
                "name": "resnet",
                "num_layers": 50
            }
        }
        yaml_path = os.path.join(tmpdirname, "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f)
        
        # Create JSON config file
        json_config = {
            "batch_size": 64,
            "learning_rate": 0.01,
            "model": {
                "name": "vgg",
                "num_layers": 16
            }
        }
        json_path = os.path.join(tmpdirname, "config.json")
        with open(json_path, "w") as f:
            json.dump(json_config, f)
        
        # Invalid extension file
        invalid_path = os.path.join(tmpdirname, "config.txt")
        with open(invalid_path, "w") as f:
            f.write("Invalid config file")
        
        yield yaml_path, json_path, invalid_path


def test_load_config(temp_config_files):
    yaml_path, json_path, invalid_path = temp_config_files
    
    # Test YAML loading
    yaml_config = load_config(yaml_path)
    assert isinstance(yaml_config, DictConfig)
    assert yaml_config.batch_size == 32
    assert yaml_config.learning_rate == 0.001
    assert yaml_config.model.name == "resnet"
    assert yaml_config.model.num_layers == 50
    
    # Test JSON loading
    json_config = load_config(json_path)
    assert isinstance(json_config, DictConfig)
    assert json_config.batch_size == 64
    assert json_config.learning_rate == 0.01
    assert json_config.model.name == "vgg"
    assert json_config.model.num_layers == 16
    
    # Test nonexistent file
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")
    
    # Test invalid file extension
    with pytest.raises(ValueError):
        load_config(invalid_path)


def test_setup_config(temp_config_files):
    yaml_path, _, _ = temp_config_files
    
    # Create a parser that doesn't have config_path parameter
    # This simulates a case where setup_config is called without config_path
    parser = argparse.ArgumentParser()
    
    # Test with just n_proc
    args = ["--n_proc", "2"]
    conf = setup_config(parser, args)
    
    assert conf.launch_config is not None
    assert conf.n_gpu == 2
    assert conf.config_path is None


def test_setup_config_with_config_path(temp_config_files):
    yaml_path, _, _ = temp_config_files
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    
    # Test with config path
    args = ["--config_path", yaml_path, "--n_proc", "2"]
    conf = setup_config(parser, args)
    
    assert conf.launch_config is not None
    assert conf.n_gpu == 2
    assert conf.config_path == yaml_path 