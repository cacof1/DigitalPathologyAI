python Application/Preprocess.py --source $1 --save_dir Preprocessing/ --patch_size 256 --step_size 256 --seg --patch --stitch 
mkdir Preprocessing/wsi
mv $1/*.svs Preprocessing/wsi
