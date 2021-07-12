import openslide,sys,glob
filelist = glob.glob(sys.argv[1]+"*.svs")

for filepath in filelist:
    wsi = openslide.open_slide(filepath)
    filename = filepath.split("/")[-1]
    
    wsi.associated_images['label'].save(filename[:-4]+'.png')
