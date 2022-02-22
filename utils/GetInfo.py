from datetime import date

def ShowTrainValTestInfo(data, config):

    target = config['DATA']['Target']

    if config['VERBOSE']['Data_Info'] and config['MODEL']['Inference'] is False:

        # Return some stats about what you're training/validating on...
        for sarcomalabel in data.train_data.coords[target].unique():
            print('Your training dataset has {}/{} patches of class {}.'.format(sum(data.train_data.coords[target] == sarcomalabel), len(data.train_data.coords[target]), sarcomalabel))
        fc = data.train_data.coords.file_id.copy()
        print('Distribution of the {} patches from the {} file_ids within the training dataset: '.format(len(fc),len(fc.unique())))
        for f in fc.unique():
            print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
        print('------------------')
        for sarcomalabel in data.val_data.coords[target].unique():
            print('Your validation dataset has {}/{} patches of class {}.'.format(sum(data.val_data.coords[target] == sarcomalabel), len(data.val_data.coords[target]), sarcomalabel))
        fc = data.val_data.coords.file_id.copy()
        print('Distribution of the {} patches from the {} file_ids within the validation dataset: '.format(len(fc),len(fc.unique())))
        for f in fc.unique():
            print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
        print('------------------')

        try:
            for sarcomalabel in data.test_data.coords[target].unique():
                print('Your test dataset has {}/{} patches of class {}.'.format(sum(data.test_data.coords[target] == sarcomalabel), len(data.test_data.coords[target]), sarcomalabel))
            fc = data.test_data.coords.file_id.copy()
            print('Distribution of the {} patches from the {} file_ids within the test dataset: '.format(len(fc),len(fc.unique())))
            for f in fc.unique():
                print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
            print('------------------')
        except:
            print('No test data.')


def format_model_name(config):

    # Generates a filename (for logging) from the configuration file.
    # This format has been validated (and designed specifically) for the sarcoma classification problem.
    # It may work with other models, or require adaptation.

    # ----------------------------------------------------------------------------------------------------------------
    # Model block
    model_block = 'empty_model'
    if config['MODEL']['Base_Model'].lower() == 'vit':
        model_block = '_d' + str(config['MODEL']['Depth']) +\
                     '_emb' + str(config['MODEL']['Emb_size']) +\
                     '_h' + str(config['MODEL']['N_Heads_ViT']) +\
                     '_subP' + str(config['DATA']['Sub_Patch_Size'])

    elif config['MODEL']['Base_Model'].lower() == 'convnet':
        pre = '_pre' if config['MODEL']['Pretrained'] is True else ''
        model_block = '_' + config['MODEL']['Backbone'] + pre +\
                     '_drop' + str(config['MODEL']['Drop_Rate'])

    elif config['MODEL']['Base_Model'].lower() == 'convnext':
        pre = 'pre' if config['MODEL']['Pretrained'] is True else ''
        model_block = '_' + pre +\
                      '_drop' + str(config['MODEL']['Drop_Rate']) +\
                     '_LS' + str(config['MODEL']['Layer_Scale']) +\
                     '_SD' + str(config['REGULARIZATION']['Stoch_Depth'])

    # ----------------------------------------------------------------------------------------------------------------
    # General model parameters block

    dimstr = ''
    for dim in range(len(config['DATA']['Dim'])):
        dimstr = dimstr + str(config['DATA']['Dim'][dim][0]) + '_'

    visstr = ''
    for vis in range(len(config['DATA']['Vis'])):
        visstr = visstr + str(config['DATA']['Vis'][dim]) + '_'

    main_block = '_dim' + dimstr +\
                 'vis' + visstr +\
                 'b' + str(config['MODEL']['Batch_Size']) +\
                 '_N' + str(config['DATA']['N_Classes']) +\
                 '_n' + str(config['DATA']['N_Per_Sample']) +\
                 '_epochs' + str(config['MODEL']['Max_Epochs']) +\
                 '_train' + str(int(100 * config['DATA']['Train_Size'])) +\
                 '_val' + str(int(100 * config['DATA']['Val_Size'])) +\
                 '_seed' + str(config['MODEL']['Random_Seed'])

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
                      '_W' + str(config['SCHEDULER']['Warmup_Epochs'])
    elif str(config['SCHEDULER']['Type']) == 'stepLR':
        sched_block = '_' + str(config['SCHEDULER']['Type']) +\
                      '_G' + str(config['SCHEDULER']['Lin_Gamma']) +\
                      '_SS' + str(config['SCHEDULER']['Lin_Step_Size'])

    # ----------------------------------------------------------------------------------------------------------------
    # Regularization block (all methods), includes CF due to label smoothing
    reg_block = '_' + str(config['MODEL']['Loss_Function']) +\
                '_LS' + str(config['REGULARIZATION']['Label_Smoothing'])

    # ----------------------------------------------------------------------------------------------------------------
    # Data Augment block (all methods)
    DA_block = '_RandAugment_n' + str(config['AUGMENTATION']['Rand_Operations']) +\
               '_M' + str(config['AUGMENTATION']['Rand_Magnitude'])

    # ----------------------------------------------------------------------------------------------------------------
    # quality control (QC) block (all methods)
    QC_block = ''
    if config['QC']['Macenko_Norm'] is True:
        QC_block = '_macenko'

    # ----------------------------------------------------------------------------------------------------------------
    # Block for moment of data acquisition
    time_block = '_' + date.today().strftime("%b-%d")

    # Append final information
    name = config['MODEL']['Base_Model'] + model_block + main_block + optim_block + sched_block + reg_block +\
        QC_block + DA_block + time_block


    if config['VERBOSE']['Data_Info']:
        print('Processing under name {}...'.format(name))

    return name
