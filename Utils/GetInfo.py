from datetime import date
import os
import numpy as np

def ShowTrainValTestInfo(data, config):

    target = config['DATA']['Label']

    # todo: reorganise this mess below.

    if config['ADVANCEDMODEL']['Inference'] is False:

        label_counter = np.zeros(len(data.train_data.tile_dataset[target].unique()))

        if config['VERBOSE']['Data_Info']:

            # Return some stats about what you're training/validating on...
            for label in data.train_data.tile_dataset[target].unique():
                npts_train = sum(data.train_data.tile_dataset[target] == label)
                print('Your training dataset has {}/{} ({:.2f}%) patches of class {}.'.format(npts_train, len(data.train_data.tile_dataset[target]), npts_train/len(data.train_data.tile_dataset[target])*100, label))
                label_counter[label] += npts_train

            fc = data.train_data.tile_dataset.SVS_ID.copy()
            print('Distribution of the {} patches from the {} file_ids within the training dataset: '.format(len(fc),len(fc.unique())))
            for f in fc.unique():
                print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
            print('------------------')
            for label in data.val_data.tile_dataset[target].unique():
                npts_valid = sum(data.val_data.tile_dataset[target] == label)
                print('Your validation dataset has {}/{} ({:.2f}%) patches of class {}.'.format(npts_valid, len(data.val_data.tile_dataset[target]), npts_valid/len(data.val_data.tile_dataset[target])*100, label))
                label_counter[label] += npts_valid

            fc = data.val_data.tile_dataset.SVS_ID.copy()
            print('Distribution of the {} patches from the {} file_ids within the validation dataset: '.format(len(fc),len(fc.unique())))
            for f in fc.unique():
                print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
            print('------------------')

            try:
                for label in data.test_data.tile_dataset[target].unique():
                    print('Your test dataset has {}/{} patches of class {}.'.format(sum(data.test_data.tile_dataset[target] == label), len(data.test_data.tile_dataset[target]), label))
                fc = data.test_data.tile_dataset.file_id.copy()
                print('Distribution of the {} patches from the {} file_ids within the test dataset: '.format(len(fc),len(fc.unique())))
                for f in fc.unique():
                    print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
                print('------------------')
            except:
                print('No test data.')

    return label_counter


def format_model_name(config):

    # Generates a filename (for logging) from the configuration file.
    # This format has been validated (and designed specifically) for the sarcoma classification problem.
    # It may work with other models, or require adaptation.

    if config['ADVANCEDMODEL']['Inference'] is False:

        # ----------------------------------------------------------------------------------------------------------------
        # Model block
        model_block = 'empty_model'
        if config['BASEMODEL']['Model'].lower() == 'vit':
            model_block = '_d' + str(config['ADVANCEDMODEL']['Depth_ViT']) +\
                         '_emb' + str(config['ADVANCEDMODEL']['Emb_size_ViT']) +\
                         '_h' + str(config['ADVANCEDMODEL']['N_Heads_ViT']) +\
                         '_subP' + str(config['DATA']['Sub_Patch_Size'])

        elif config['BASEMODEL']['Model'].lower() == 'convnet':
            pre = '_pre' if config['ADVANCEDMODEL']['Pretrained'] is True else ''
            model_block = '_' + config['BASEMODEL']['Backbone'] + pre +\
                         '_drop' + str(config['ADVANCEDMODEL']['Drop_Rate'])

        elif config['BASEMODEL']['Model'].lower() == 'convnext':
            pre = 'pre' if config['ADVANCEDMODEL']['Pretrained'] is True else ''
            model_block = '_' + pre +\
                          '_drop' + str(config['ADVANCEDMODEL']['Drop_Rate']) +\
                         '_LS' + str(config['ADVANCEDMODEL']['Layer_Scale']) +\
                         '_SD' + str(config['REGULARIZATION']['Stoch_Depth'])

        # ----------------------------------------------------------------------------------------------------------------
        # General model parameters block

        dimstr = ''
        for dim in range(len(config['BASEMODEL']['Patch_Size'])):
            dimstr = dimstr + str(config['BASEMODEL']['Patch_Size'][dim][0]) + '_'

        visstr = ''
        for vis in range(len(config['BASEMODEL']['Vis'])):
            visstr = visstr + str(config['BASEMODEL']['Vis'][dim]) + '_'

        main_block = '_dim' + dimstr +\
                     'vis' + visstr +\
                     'b' + str(config['BASEMODEL']['Batch_Size']) +\
                     '_N' + str(config['DATA']['N_Classes']) +\
                     '_n' + str(config['DATA']['N_Per_Sample']) +\
                     '_epochs' + str(config['ADVANCEDMODEL']['Max_Epochs']) +\
                     '_train' + str(int(100 * config['DATA']['Train_Size'])) +\
                     '_val' + str(int(100 * config['DATA']['Val_Size'])) +\
                     '_seed' + str(config['ADVANCEDMODEL']['Random_Seed'])

        # ----------------------------------------------------------------------------------------------------------------
        # Optimisation block (all methods)
        optim_block = '_' + str(config['OPTIMIZER']['Algorithm']) +\
                      '_lr' + str(config['OPTIMIZER']['lr']) +\
                      '_eps' + str(config['OPTIMIZER']['eps']) +\
                      '_WD' + str(config['REGULARIZATION']['Weight_Decay'])

        # ----------------------------------------------------------------------------------------------------------------
        # Scheduler block (all methods)
        sched_block = 'empty_scheduler'
        if str(config['SCHEDULER']['Type']) == 'cosine_warmup':
            sched_block = '_' + str(config['SCHEDULER']['Type']) +\
                          '_W' + str(config['SCHEDULER']['Cos_Warmup_Epochs'])
        elif str(config['SCHEDULER']['Type']) == 'stepLR':
            sched_block = '_' + str(config['SCHEDULER']['Type']) +\
                          '_G' + str(config['SCHEDULER']['Lin_Gamma']) +\
                          '_SS' + str(config['SCHEDULER']['Lin_Step_Size'])

        # ----------------------------------------------------------------------------------------------------------------
        # Regularization block (all methods), includes CF due to label smoothing
        reg_block = '_' + str(config['BASEMODEL']['Loss_Function']) +\
                    '_LS' + str(config['REGULARIZATION']['Label_Smoothing'])

        # ----------------------------------------------------------------------------------------------------------------
        # Data Augment block (all methods)
        DA_block = '_RandAugment_n' + str(config['AUGMENTATION']['Rand_Operations']) +\
                   '_M' + str(config['AUGMENTATION']['Rand_Magnitude'])

        # ----------------------------------------------------------------------------------------------------------------
        # quality control (QC) block (all methods)
        QC_block = ''
        if 'Colour_Norm_File' in config['NORMALIZATION']:
            QC_block = '_macenko'

        # ----------------------------------------------------------------------------------------------------------------
        # Block for moment of data acquisition
        time_block = '_' + date.today().strftime("%b-%d")

        # Append final information
        name = config['BASEMODEL']['Model'] + model_block + main_block + optim_block + sched_block + reg_block +\
            QC_block + DA_block + time_block


        if config['VERBOSE']['Data_Info']:
            print('Processing under name {}...'.format(name))

    else:

        # Create a name using the pre-trained model path (without its .ckt extension)
        name = 'Inference_using_' + config['CHECKPOINT']['Model_Save_Path'].split(os.path.sep)[-1][:-5]

    return name
