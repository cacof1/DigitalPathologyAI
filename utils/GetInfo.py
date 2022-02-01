from datetime import date

def ShowTrainValTestInfo(data, config):

    target = config['DATA']['target']

    if config['VERBOSE']['data_info'] and config['MODEL']['inference'] is False:

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

    # transformer params
    if config['MODEL']['BaseModel'] == 'Transformer':
        vit_info = '_emb' + str(config['MODEL']['emb_size']) + '_nheads' + str(config['MODEL']['n_heads_vit']) + '_depth' + str(config['MODEL']['depth'])
    else:
        vit_info = ''


    model_name = config['MODEL']['Backbone'] + '_pre' if config['MODEL']['Pretrained'] is True else config['MODEL'][
        'Backbone'] + vit_info
    model_name = model_name + '_drop' + str(config['MODEL']['Drop_Rate'])

    dimstr = ''
    for dim in range(len(config['DATA']['dim'])):
        dimstr = dimstr + str(config['DATA']['dim'][dim][0]) + '_'

    name = config['MODEL']['BaseModel'] + model_name + '_lr' + str(config['OPTIMIZER']['lr']) + '_dim' + dimstr +\
           'batch' + str(config['MODEL']['Batch_Size']) + '_N' + str(config['DATA']['n_classes']) + '_n' + \
           str(config['DATA']['n_per_sample']) + '_epochs' + str(config['MODEL']['Max_Epochs']) + '_train' +\
           str(int(100 * config['DATA']['train_size'])) + '_val' + str(int(100 * config['DATA']['val_size'])) +\
           '_loss' + config['MODEL']['loss_function'] + '_seed' + str(config['MODEL']['RANDOM_SEED']) +\
           '_' + date.today().strftime("%b-%d")

    if config['VERBOSE']['data_info']:
        print('Processing under name {}...'.format(name))

    return name
