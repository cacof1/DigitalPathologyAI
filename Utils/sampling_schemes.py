import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sample_n_per_sample_per_label_and_equalize(tile_dataset, target=None, n_per_sample=np.Inf, train_size=None,
                                               test_size=None, verbose=True):
    # Data sampling scheme
    # Consider n_WSI individual slides, n_labels labels, and n_per_sample an integer

    # ** The total number of patches to be extracted per WSI is ideally n_labels x n_per_sample **

    # The number of overall patches assigned to a label will be a maximum value of n_per_sample*n_WSI. This
    # means ideally that n_per_sample patches will be extracted for each WSI. However, if some WSI have less than
    # n_per_sample patches, the missing number of patches will be re-attributed to other WSIs.

    # This is appropriate if you have large heterogeneity in your data (high variability in the number of patches per
    # label across WSIs).

    all_ids = tile_dataset['SVS_ID'].unique()
    all_labels = tile_dataset[target].unique()
    final_contribution = np.zeros((len(all_labels), len(all_ids)))

    for li in range(len(all_labels)):
        cl = all_labels[li]
        df_per_label = tile_dataset[tile_dataset[target] == cl]
        n_labels = len(df_per_label)

        # Adding samples by chunks of n_per_sample/10 until we have enough!
        increment = 200
        max_pts = np.min([n_labels, n_per_sample*len(all_ids)])
        while True:

            if sum(final_contribution[li, :]) >= max_pts:
                break

            for idi in range(len(all_ids)):
                cid = all_ids[idi]
                df_per_label_per_svs = df_per_label[df_per_label['SVS_ID'] == cid]
                npatches_in_sample = len(df_per_label_per_svs)

                if (final_contribution[li, idi] + increment) <= npatches_in_sample:
                    final_contribution[li, idi] += increment

                elif ((final_contribution[li, idi] + increment) > npatches_in_sample) & (final_contribution[li, idi] <= npatches_in_sample):
                    final_contribution[li, idi] = npatches_in_sample

        if verbose:
            print('{} patches used on a total of {} for label # {}'.format(np.sum(final_contribution[li, :]).astype(int), n_labels, cl))

    # The output of the above is the "final_contribution" array, of size n_labels x n_WSI. Each element provides
    # the number of patches that will be for each label for each WSI.

    # Patch sampling is achieved with the following:
    df_list = []
    grouped = tile_dataset.groupby('SVS_ID')
    for current_fileid, group in grouped:
        grouped2 = group.groupby(target)
        for current_label, group2 in grouped2:
            yy = np.argwhere(all_ids == current_fileid)[0][0]
            xx = np.argwhere(all_labels == current_label)[0][0]
            df_list.append(group2.sample(n=final_contribution[xx, yy].astype(int), replace=False))

    tile_dataset_sampled = pd.concat(df_list)

    # Now we need to create training and validation sets. In this scheme we will split accoridng to the train/valid
    # fractions each label, and then stratify the data to have similar occurences.
    X = np.arange(len(tile_dataset_sampled))
    y = tile_dataset_sampled[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, stratify=y)

    return tile_dataset_sampled.iloc[X_train], tile_dataset_sampled.iloc[X_test]

def sample_N_per_WSI(tile_dataset, n_per_sample=np.Inf):
    # Consider n_per_sample, an integer and n_patches_i, the total number of patches for a WSI of index i

    # Simple sampler: randomly sample min(n_per_sample,n_patches_i) patches for each WSI of index i.
    # This is appropriate when you have a single class per WSI, or if the classes are balanced within each WSI.

    value_counts = tile_dataset.SVS_ID.value_counts()
    fn_for_sampling = value_counts[value_counts > n_per_sample].index
    df1 = tile_dataset[tile_dataset['SVS_ID'].isin(fn_for_sampling)].groupby("SVS_ID").sample(n=n_per_sample,
                                                                                              replace=False)

    if fn_for_sampling.shape != value_counts.shape:  # if some datasets have less than n_per_sample
        df2 = tile_dataset[~tile_dataset['SVS_ID'].isin(fn_for_sampling)].groupby("SVS_ID").sample(frac=1)
        return pd.concat([df1, df2])
    else:
        return df1

