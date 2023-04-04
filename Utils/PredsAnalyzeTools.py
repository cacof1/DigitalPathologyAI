import numpy as np
import pandas as pd
import openslide
import matplotlib.pyplot as plt


def Preds2Results(Preds,
                  slide_dataset,
                  batch_size,
                  Detection_Path='/home/dgs2/data/DigitalPathologyAI/MitoticDetection/DetectionResults/',
                  threshold=0.5,
                  label_name='num_objs',
                  save_name='detected'):
    detected_coords = []
    detected_masks = []
    detected_scores = []
    detected_boxes = []
    tumour_probs = []
    gt_labels = []

    for i in range(slide_dataset.shape[0]):
        num_of_batch = int(i / batch_size)
        prediction = Preds[num_of_batch][int(i % batch_size)]
        boxes = prediction['boxes'].cpu().detach().numpy()
        labels = prediction['labels'].cpu().detach().numpy()
        scores = prediction['scores'].cpu().detach().numpy()
        masks = prediction['masks'].cpu().detach().numpy()
        top_left = (slide_dataset.coords_x[i], slide_dataset.coords_y[i])
        gt_label = slide_dataset[label_name][i]

        if len(scores) == 0:
            label = 0
            max_score = 0
        else:
            max_score = int(max(scores) * 100) / 100
            label = labels[np.argmax(scores)]
            box = boxes[np.argmax(scores)]
            mask = np.squeeze(masks[np.argmax(scores)])

        if label == 1:
            if max_score > threshold:
                detected_coords.append(top_left)
                detected_masks.append(mask)
                detected_scores.append(max_score)
                detected_boxes.append(box)
                tumour_probs.append(slide_dataset['prob_tissue_type_Tumour'][i])
                gt_labels.append(gt_label)

        if i % 5000 == 0:
            print('{}/{} tiles processed'.format(i, slide_dataset.shape[0]))

    df = pd.DataFrame()

    detected_coords = np.array(detected_coords)
    detected_masks = np.array(detected_masks)
    detected_scores = np.array(detected_scores)
    detected_boxes = np.array(detected_boxes)
    tumour_probs = np.array(tumour_probs)

    df['scores'] = np.array(detected_scores)
    df['coords_x'] = np.array(detected_coords)[:, 0]
    df['coords_y'] = np.array(detected_coords)[:, 1]
    df['xmin'] = np.array(detected_boxes)[:, 0]
    df['xmax'] = np.array(detected_boxes)[:, 2]
    df['ymin'] = np.array(detected_boxes)[:, 1]
    df['ymax'] = np.array(detected_boxes)[:, 3]
    df['prob_tissue_type_Tumour'] = np.array(tumour_probs)
    df['gt_label'] = np.array(gt_labels)

    df.to_csv(Detection_Path + '{}_{}_coords.csv'.format(slide_dataset['id_external'][0], save_name), index=False)
    np.save(Detection_Path + '{}_{}_masks.npy'.format(slide_dataset['id_external'][0], save_name), detected_masks)

    print('Number of Mitotic Proposals: {}'.format(df.shape[0]))

    return df, detected_masks

def CombineGtsandPreds(SVS_list, tiledataset,
                       Detection_Path='/home/dgs2/data/DigitalPathologyAI/MitoticDetection/DetectionResults/'):

    df_list = []
    for SVS_ID in SVS_list:
        slidedataset = tiledataset[tiledataset['SVS_ID'] == SVS_ID]
        slidedataset = slidedataset.loc[:,
                       ['coords_x', 'coords_y', 'SVS_ID', 'index', 'num_objs', 'prob_tissue_type_Tumour']]
        slidedataset.rename(columns={"num_objs": "gt_label"}, inplace=True)
        slidedataset['scores'] = [1.0]*slidedataset.shape[0]
        detection_df = pd.read_csv(Detection_Path + '{}_detected_coords.csv'.format(SVS_ID))
        detection_df = detection_df[detection_df['scores']>0.8]
        detection_df = detection_df.loc[:,['scores','coords_x', 'coords_y', 'gt_label', 'prob_tissue_type_Tumour']]
        detection_df['SVS_ID'] = [SVS_ID] * detection_df.shape[0]
        detection_df = detection_df.reindex(columns=slidedataset.columns)
        detection_df['index'] = detection_df.index

        combined_df = pd.concat(
            [slidedataset[slidedataset['gt_label'] == 1], detection_df[detection_df['gt_label'] == 0]],
            axis=0).reset_index(drop=True)
        #combined_df.to_csv(Detection_Path + 'combined_dataset_{}.csv'.format(SVS_ID), index=False)
        #print('combined_dataset_{}.csv Saved'.format(SVS_ID))

        df_list.append(combined_df)

    df_all = pd.concat(df_list, axis=0)
    df_all.to_csv(Detection_Path + 'combined_dataset.csv', index=False)
    return df_all

'''
SVS_list = ['6a_210222','5a_165116','10a_210222','46a_111900','10a_170207','44a_112335','70a_040622','22-07-13_85a','53a_040622','42a_111222',
            '43a_112516','45a_112105','64a_040622','65a_040622','22-07-13_83a','67a_040622','4a_151121','62a_040622','1a_180122','1f_180122','3a_165627',
            '7a_210222','8a_210222','22-07-13_84a','4a_101221','9a_210222','22-07-13_75a','7a_151121','8a_151121','4a_030122','5a_030122','6a_030122',
            '7a_030122','1b_180122','1g_180122','1a_165925','55a_040622','56a_040622','22-07-13_72a','22-07-13_89a','22-07-13_91a','9a_170029','2a_210222','61a_040622',
            '9a_151121','3a_101221','8a_030122','7a_164852','4a_210222','59a_040622','22-07-13_82a','1a_101221','58a_040622','22-07-13_81a','22-07-13_88a','6a_101221',
            '4a_165444', '51a_040622', '22-07-13_92a', '3a_210222', '22-07-13_98a', '10a_030122', '49a_111407',
            '22-07-13_97a', '48a_111529', '52a_040622', '1a_151121','3a_030122',
             '1h_180122', '1j_180122', '22-07-13_90a', '57a_040622', '22-07-13_73a', '22-07-13_77a', '22-07-13_87a',
             '22-07-13_99a', '8a_101221', '1c_180122', '1e_180122', '1i_180122',
             '2a_165745', '66a_040622', '22-07-13_71a', '22-07-13_76a', '22-07-13_93a', '22-07-13_94a', '22-07-13_95a',
             '22-07-13_96a', '22-07-13_79a', '8a_165002', '50a_111959',
             '54a_040622', '2a_101221', '22-07-13_86a', '9a_030122','6a_165331','6a_151121','69a_040622','68a_040622','7a_101221','63a_040622','41a_111313','60a_040622','22-07-13_80a','47a_111639','22-07-13_74a',]

tiledataset = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/all_tiles_0208.csv')
df_all = CombineGtsandPreds(SVS_list, tiledataset)
df_all.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/combined_dataset.csv', index=False)
'''