## First remove contour
#python Inference/Preprocess.py --source $1 --save_dir ./ --patch_size 256 --step_size 256 --seg --patch --stitch 

##Then tag those that are tumours
python Inference/Classify_tumor.py $2 wsi patches PretrainedModel/epoch\=02-val_acc\=1.00_tumour.ckpt
