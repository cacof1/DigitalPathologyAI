from Utils import npyExportTools
import numpy as np
import glob
import toml,sys
config   = toml.load(sys.argv[1])
cur_basemodel_str = npyExportTools.basemodel_to_str(config)
for npy_path in glob.glob(sys.argv[2]+"*.npy"):
    header,df  =  next(iter(np.load(npy_path, allow_pickle=True).item().values()))
    npy_dict = {}
    npy_dict[cur_basemodel_str] = [header, df]
    print(npy_dict)
    np.save(npy_path, npy_dict)


