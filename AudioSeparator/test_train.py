# python3 test_train.py


#########################################
#///////////////////////////////////////#
#########################################


import unittest
import argparse, yaml
from train import *
from model import *
from data import *


#########################################
#///////////////////////////////////////#
#########################################


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def tearDown(self):
        pass

    def test_fuss_train_rha(self):
        configs = get_configs(yaml_test_overide='configs/rha_train_default.yaml')

        # Get fuss dataset
        dataset = getData(configs,self.AUTOTUNE,"eval")
        for sample in dataset.take(1):
            mix_data, source = sample

        self.assertEqual(mix_data.shape,(configs['batch_size'],79872))
        self.assertEqual(source.shape,(configs['batch_size'],4,79872))

        # Get model
        model = getModel(configs,print_model=False)

        # Train
        history = model.fit(mix_data, source, epochs=configs['epochs'],batch_size=configs['batch_size'],verbose=0)

    def test_fuss_train_dm_rha(self):
        configs = get_configs(yaml_test_overide='configs/rha_train_dm_default.yaml')

        # Get fuss dataset
        dataset = getData(configs,self.AUTOTUNE,"train")
        for sample in dataset.take(1):
            mix_data, source = sample

        self.assertEqual(mix_data.shape,(configs['batch_size'],79872))
        self.assertEqual(source.shape,(configs['batch_size'],4,79872))

        # Get model
        model = getModel(configs,print_model=False)

        # Train
        history = model.fit(mix_data, source, epochs=configs['epochs'],batch_size=configs['batch_size'],verbose=0)


#########################################
#///////////////////////////////////////#
#########################################


if __name__ == "__main__":
    unittest.main()
