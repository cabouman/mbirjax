"""
Tests that each model instance owns a private logger.

Two models of the same class must not share one logger object: a shared logger means
concurrent models rebuild each other's handlers, and one thread can close a log file while
another thread is still writing to it.
"""
import logging
import os
import tempfile
import unittest

import numpy as np
import mbirjax as mj


def _make_model(num_views=8, num_det_rows=6, num_det_channels=10):
    """A tiny parallel-beam model; nothing here is reconstructed, only logged."""
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    return mj.ParallelBeamModel((num_views, num_det_rows, num_det_channels), angles)


class TestInstanceLogger(unittest.TestCase):
    """Per-instance logger isolation."""

    def test_loggers_are_distinct_per_instance(self):
        print('Testing that two models of one class get distinct loggers')
        model_a, model_b = _make_model(), _make_model()
        model_a.setup_logger(logfile_path=None, print_logs=False)
        model_b.setup_logger(logfile_path=None, print_logs=False)

        self.assertIsNot(model_a.logger, model_b.logger)
        self.assertIsNot(model_a.log_buffer, model_b.log_buffer)

        model_a.logger.info('message for a')
        self.assertIn('message for a', model_a.log_buffer.getvalue())
        self.assertNotIn('message for a', model_b.log_buffer.getvalue())

    def test_logger_is_outside_the_global_registry(self):
        """The logger must be unreachable by name, so no second instance can pick it up."""
        print('Testing that the instance logger is not in the logging registry')
        model = _make_model()
        model.setup_logger(logfile_path=None, print_logs=False)
        self.assertNotIn(model.logger.name, logging.Logger.manager.loggerDict)

    def test_setup_logger_reuses_the_instance_logger(self):
        """Repeated setup must reconfigure the same object, closing the handlers it replaces."""
        print('Testing that repeated setup_logger closes the old handlers')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = _make_model()
            first_path = os.path.join(tmp_dir, 'first.log')
            model.setup_logger(logfile_path=first_path, print_logs=False)
            logger_object = model.logger
            old_handlers = [h for h in logger_object.handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(old_handlers), 1)

            model.setup_logger(logfile_path=os.path.join(tmp_dir, 'second.log'), print_logs=False)
            self.assertIs(model.logger, logger_object)
            self.assertTrue(all(h.stream is None or h.stream.closed for h in old_handlers))

    def test_second_model_setup_does_not_disturb_the_first(self):
        """The race being fixed: building model B's handlers must leave A's file handler open."""
        print('Testing that setting up a second model leaves the first model logging')
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_a = os.path.join(tmp_dir, 'a.log')
            model_a = _make_model()
            model_a.setup_logger(logfile_path=path_a, print_logs=False)

            model_b = _make_model()
            model_b.setup_logger(logfile_path=os.path.join(tmp_dir, 'b.log'), print_logs=False)

            model_a.logger.info('written after b was set up')
            for handler in model_a.logger.handlers:
                handler.flush()
            with open(path_a) as f:
                self.assertIn('written after b was set up', f.read())


if __name__ == '__main__':
    unittest.main()
