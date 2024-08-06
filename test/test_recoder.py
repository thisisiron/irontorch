import os
import shutil
import argparse
import logging
import unittest
from unittest import TestCase, mock
from irontorch.recoder import get_logger, Logger, WandB 
import wandb
import irontorch.distributed as dist


class TestLogging(TestCase):

    def setUp(self):
        self.test_dir = './tmp'
        os.makedirs(self.test_dir, exist_ok=True)
        self.logger_name = f'test_logger_{id(self)}'

    # def tearDown(self):
    #     shutil.rmtree(self.test_dir)
    #     logging.shutdown()
    #     logging.root.handlers.clear()
    #     for handler in logging.root.handlers[:]:
    #         logging.root.removeHandler(handler)
    #         handler.close()

    def test_get_logger(self):
        logger = get_logger(self.test_dir, name=self.logger_name, distributed_rank=0)
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

    def test_logger_rich(self):
        print('Running Rich mode')
        logger = Logger(self.test_dir, rank=0, mode='rich', name=self.logger_name)
        logger.log(1, accuracy=0.95, loss=0.05)
        
        # Check if the logs are printed to stdout
        with self.assertLogs(self.logger_name, level='INFO') as log:
            logger.log(2, accuracy=0.96, loss=0.04)
            self.assertIn(f'INFO:{self.logger_name}:step: 2 | accuracy: 0.96 | loss: 0.04', log.output)

    def test_logger_color(self):
        print('Running Color mode')
        logger = Logger(self.test_dir, rank=0, mode='color', name=self.logger_name)
        logger.log(1, accuracy=0.95, loss=0.05)
        
        # Check if the logs are printed to stdout
        with self.assertLogs(self.logger_name, level='INFO') as log:
            logger.log(2, accuracy=0.96, loss=0.04)
            self.assertIn(f'INFO:{self.logger_name}:step: 2 | accuracy: 0.96 | loss: 0.04', log.output)

    def test_logger_plain(self):
        print('Running Plain mode')
        logger = Logger(self.test_dir, rank=0, mode='plain', name=self.logger_name)
        logger.log(1, accuracy=0.95, loss=0.05)
        
        # Check if the logs are printed to stdout
        with self.assertLogs(self.logger_name, level='INFO') as log:
            logger.log(2, accuracy=0.96, loss=0.04)
            self.assertIn(f'INFO:{self.logger_name}:step: 2 | accuracy: 0.96 | loss: 0.04', log.output)
        del logger

    @unittest.skipUnless('RUN_WANDB_TEST' in os.environ, "Skipping WandB test")
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
    parser = argparse.ArgumentParser(description='Run logging tests')
    parser.add_argument('--run_wandb', action='store_true', help='Run WandB test')
    args = parser.parse_args()

    if args.run_wandb:
        os.environ['RUN_WANDB_TEST'] = '1'

    unittest.main(argv=[''], verbosity=2, exit=False)
