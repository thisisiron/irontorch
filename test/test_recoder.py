import os
import shutil
import tempfile
import logging
from unittest import TestCase, mock
from irontorch.recoder import get_logger, Logger, WandB 
import wandb
import irontorch.distributed as dist


class TestLogging(TestCase):

    def setUp(self):
        self.test_dir = './tmp'
        os.makedirs(self.test_dir, exist_ok=True)
        self.logger_name = 'test_logger'

    # def tearDown(self):
    #     shutil.rmtree(self.test_dir)

    def test_get_logger(self):
        logger = get_logger(self.test_dir, self.logger_name, distributed_rank=0)
        logger.debug('This is a debug message')
        logger.info('This is an info message')
        logger.warning('This is a warning message')
        
        log_file_path = os.path.join(self.test_dir, 'log.txt')
        self.assertTrue(os.path.exists(log_file_path))
        
        with open(log_file_path, 'r') as log_file:
            log_contents = log_file.read()
            self.assertIn('This is a debug message', log_contents)
            self.assertIn('This is an info message', log_contents)
            self.assertIn('This is a warning message', log_contents)

    def test_logger_class(self):
        logger = Logger(self.test_dir, rank=0, mode='rich')
        logger.log(1, accuracy=0.95, loss=0.05)
        
        # Check if the logs are printed to stdout
        with self.assertLogs('main', level='INFO') as log:
            logger.log(2, accuracy=0.96, loss=0.04)
            self.assertIn('INFO:main:step: 2| accuracy: 0.96| loss: 0.04', log.output)

    def test_wandb_class(self):
        # Note: Make sure to set up a W&B project and configure your W&B API key.
        wandb_logger = WandB(
            project='test_project',
            group='test_group',
            name='test_run',
            notes='test_notes',
            resume='allow',
            tags=['test'],
            id='test_id'
        )

        wandb_logger.log(1, accuracy=0.95, loss=0.05)

        del wandb_logger


if __name__ == '__main__':
    import unittest
    unittest.main()
