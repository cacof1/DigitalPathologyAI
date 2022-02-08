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

    print('TO REBUILD...')

    # Generates a filename (for logging) from the configuration file.
    # This format has been validated (and designed specifically) for the sarcoma classification problem.
    # It may work with other models, or require adaptation.

    # transformer params
    if config['MODEL']['Base_Model'] == 'Transformer':
        vit_info = '_emb' + str(config['MODEL']['Emb_Size']) + '_nheads' + str(config['MODEL']['n_heads_vit']) + '_depth' + str(config['MODEL']['depth'])
    else:
        vit_info = ''


    model_name = config['MODEL']['Backbone'] + '_pre' if config['MODEL']['Pretrained'] is True else config['MODEL'][
        'Backbone'] + vit_info
    model_name = model_name + '_drop' + str(config['MODEL']['Drop_Rate'])

    dimstr = ''
    for dim in range(len(config['DATA']['Dim'])):
        dimstr = dimstr + str(config['DATA']['Dim'][dim][0]) + '_'

    name = config['MODEL']['Base_Model'] + model_name + '_lr' + str(config['OPTIMIZER']['lr']) + '_dim' + dimstr +\
           'batch' + str(config['MODEL']['Batch_Size']) + '_N' + str(config['DATA']['N_Classes']) + '_n' + \
           str(config['DATA']['N_Per_Sample']) + '_epochs' + str(config['MODEL']['Max_Epochs']) + '_train' +\
           str(int(100 * config['DATA']['Train_Size'])) + '_val' + str(int(100 * config['DATA']['Val_Size'])) +\
           '_loss' + config['MODEL']['Loss_Function'] + '_seed' + str(config['MODEL']['Random_Seed']) +\
           '_' + date.today().strftime("%b-%d")

    if config['VERBOSE']['Data_Info']:
        print('Processing under name {}...'.format(name))

    return name
