from PreProcessor import PreProcessor
import toml

config = toml.load('./config_files/tumour_identification_training_ubuntu.ini')
#config = toml.load('./config_files/tumour_identification_training_mac.ini')

preprocess = PreProcessor(config)
preprocess.preprocess_WSI()

