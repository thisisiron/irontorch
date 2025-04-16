import unittest
# Removed unused import: import torch
# Removed unused import: import torch.multiprocessing as mp
from torch.distributed.launcher.api import LaunchConfig
from irontorch.distributed.launch import run, elastic_worker
# Removed unused import: from irontorch.distributed.parser import setup_config
from unittest.mock import patch, MagicMock


class TestLaunchFunctions(unittest.TestCase):

    def setUp(self):
        self.fn = MagicMock()
        self.args = (1, 2, 3)
        self.conf = MagicMock()
        self.conf.n_gpu = 2
        self.conf.launch_config = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=2, run_id="test"
        )

    @patch("irontorch.distributed.launch.elastic_launch")
    def test_run_with_multiple_gpus(self, mock_elastic_launch):
        run(self.fn, self.conf, self.args)
        mock_elastic_launch.assert_called_once_with(
            config=self.conf.launch_config, entrypoint=elastic_worker
        )
        mock_elastic_launch.return_value.assert_called_once_with(
            self.fn, self.args, self.conf.n_gpu
        )

    def test_run_with_single_gpu(self):
        self.conf.n_gpu = 1
        run(self.fn, self.conf, self.args)
        self.fn.assert_called_once_with(*self.args)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.distributed.init_process_group")
    @patch("irontorch.distributed.launch.dist.synchronize")
    @patch("irontorch.distributed.launch.dist.create_local_process_group")
    @patch("torch.cuda.set_device")
    def test_elastic_worker(
        self,
        mock_set_device,
        mock_create_local_process_group,
        mock_synchronize,
        mock_init_process_group,
        mock_device_count,
        mock_is_available,
    ):
        with patch.dict("os.environ", {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2"}):
            elastic_worker(self.fn, self.args, 2)
            mock_init_process_group.assert_called_once_with(backend="NCCL")
            mock_synchronize.assert_called_once()
            mock_set_device.assert_called_once_with(0)
            mock_create_local_process_group.assert_called_once_with(2)
            self.fn.assert_called_once_with(*self.args)


if __name__ == "__main__":
    unittest.main()
