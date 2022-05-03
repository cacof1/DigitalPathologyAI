from PreProcessor import PreProcessor
import toml

config = toml.load('./config_files/tumour_identification_training_ubuntu.ini')
#config = toml.load('./config_files/tumour_identification_testing_ubuntu.ini')
# config = toml.load('./config_files/tumour_identification_testing_zhuoyan_ubuntu.ini')

preprocess = PreProcessor(config)

preprocess.preprocess_WSI()
