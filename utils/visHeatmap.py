from wsi_core.WholeSlideImage import WholeSlideImage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

patient_id = sys.argv[1]
filename   = sys.argv[2] 
wsi_object = WholeSlideImage(patient_id)
df         = pd.read_csv(filename,index_col=0)
preds      = (np.array(df["tumour_label"]))*100
coords     = np.array(df[["coords_x","coords_y"]])

heatmap = wsi_object.visHeatmap(preds, coords, vis_level=-1, segment=False)
plt.imshow(heatmap)
plt.show()
