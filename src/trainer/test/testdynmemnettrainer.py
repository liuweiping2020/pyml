# -*- coding: utf-8 -*-
import unittest

from trainer.dynmemnettrainer import DynMemNetTrainer


class TestDynMemNetTrainer(unittest.TestCase):
    def testTrainer(self):
        dynMemNetTrainer=DynMemNetTrainer()
        dynMemNetTrainer.run()
        pass
