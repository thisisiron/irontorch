import pytest
import os
import torch
from unittest.mock import patch, MagicMock

from irontorch.distributed.launch import set_omp_threads, run, elastic_worker


def test_set_omp_threads():
    # Save the original environment variable if it exists
    orig_value = os.environ.get("OMP_NUM_THREADS", None)
    
    try:
        # Remove OMP_NUM_THREADS if it exists
        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]
        
        # Call the function and check if it sets the environment variable
        set_omp_threads()
        assert os.environ["OMP_NUM_THREADS"] == "1"
        
        # Test that it doesn't change the value if it's already set
        os.environ["OMP_NUM_THREADS"] = "4"
        set_omp_threads()
        assert os.environ["OMP_NUM_THREADS"] == "4"
    
    finally:
        # Restore the original value or remove it
        if orig_value is not None:
            os.environ["OMP_NUM_THREADS"] = orig_value
        elif "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]


@patch('irontorch.distributed.launch.elastic_launch')
def test_run_single_process(mock_elastic_launch):
    # Create a mock for elastic_launch
    mock_launcher = MagicMock()
    mock_elastic_launch.return_value = mock_launcher
    
    # Create a simple function and check if it's called
    fn_called = False
    
    def test_fn():
        nonlocal fn_called
        fn_called = True
    
    # Create mock conf with n_gpu = 1
    conf = MagicMock()
    conf.n_gpu = 1
    conf.launch_config = MagicMock()
    
    # Run with a single process
    run(test_fn, conf)
    
    # Check that the function was called directly
    assert fn_called is True
    
    # Check that elastic_launch was not called
    mock_elastic_launch.assert_not_called()


@patch('irontorch.distributed.launch.elastic_launch')
def test_run_multi_process(mock_elastic_launch):
    # Create a mock for elastic_launch
    mock_launcher = MagicMock()
    mock_elastic_launch.return_value = mock_launcher
    
    # Create a simple function
    def test_fn(arg1, arg2):
        return arg1 + arg2
    
    # Create mock conf with n_gpu > 1
    conf = MagicMock()
    conf.n_gpu = 2
    conf.launch_config = MagicMock()
    
    # Run with multiple processes
    run(test_fn, conf, args=(1, 2))
    
    # Check that elastic_launch was called with the right parameters
    # The test should match the actual implementation, which passes the function directly
    mock_elastic_launch.assert_called_once_with(
        config=conf.launch_config, 
        entrypoint=elastic_worker  # Pass the actual function, not a string
    )
    
    # Check that the mock launcher was called with the right parameters
    mock_launcher.assert_called_once_with(test_fn, (1, 2), 2) 