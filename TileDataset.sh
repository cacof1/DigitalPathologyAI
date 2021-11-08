## Clear background stuff
rm -rf Preprocessing
mkdir Preprocessing
## First remove contour
python Inference/Preprocess.py --source $1 --save_dir Preprocessing/ --patch_size 256 --step_size 256 --seg --patch --stitch 
mkdir Preprocessing/wsi
mv $1/*.svs Preprocessing/wsi

##Then tag those that are tumours
python Inference/Classify_tumor.py Preprocessing PretrainedModel/epoch\=02-val_acc\=1.00_tumour.ckpt


