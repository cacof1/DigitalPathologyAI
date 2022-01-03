
def ShowTrainValTestInfo(coords_file, data):

    # Return some stats about what you're training/validating on...
    print('------------------')
    fc = coords_file.file_id.copy()
    print('Distribution of the {} patches from the {} file_ids within the entire dataset: '.format(len(fc),len(fc.unique())))
    for f in fc.unique():
        print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
    print('------------------')
    for sarcomalabel in data.train_data.coords['sarcoma_label'].unique():
        print('Your training dataset has {}/{} patches of class {}.'.format(sum(data.train_data.coords['sarcoma_label'] == sarcomalabel), len(data.train_data.coords['sarcoma_label']), sarcomalabel))
    fc = data.train_data.coords.file_id.copy()
    print('Distribution of the {} patches from the {} file_ids within the training dataset: '.format(len(fc),len(fc.unique())))
    for f in fc.unique():
        print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
    print('------------------')
    for sarcomalabel in data.val_data.coords['sarcoma_label'].unique():
        print('Your validation dataset has {}/{} patches of class {}.'.format(sum(data.val_data.coords['sarcoma_label'] == sarcomalabel), len(data.val_data.coords['sarcoma_label']), sarcomalabel))
    fc = data.val_data.coords.file_id.copy()
    print('Distribution of the {} patches from the {} file_ids within the validation dataset: '.format(len(fc),len(fc.unique())))
    for f in fc.unique():
        print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
    print('------------------')
    for sarcomalabel in data.test_data.coords['sarcoma_label'].unique():
        print('Your test dataset has {}/{} patches of class {}.'.format(sum(data.test_data.coords['sarcoma_label'] == sarcomalabel), len(data.test_data.coords['sarcoma_label']), sarcomalabel))
    fc = data.test_data.coords.file_id.copy()
    print('Distribution of the {} patches from the {} file_ids within the test dataset: '.format(len(fc),len(fc.unique())))
    for f in fc.unique():
        print('{} = {}/{} = {:.2f}%, '.format(f, sum(fc == f), len(fc), 100*sum(fc == f)/len(fc)))
    print('------------------')