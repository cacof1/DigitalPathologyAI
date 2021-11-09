import sys
import numpy as np
import matplotlib.pyplot as plt
from wsi_core.WholeSlideImage import WholeSlideImage
data = []
def onclick(event):
    counter = np.zeros((4,4))
    global image_x, image_y
    image_x, image_y = int(event.xdata), int(event.ydata)
    ix = np.floor(image_x/256).astype(np.int)
    iy = np.floor(image_y/256).astype(np.int)
    counter[ix,iy] += 1
    patch[image_y-10:image_y+10, image_x-10:image_x + 10 ] = [0,0,0,0]
    #patch[iy*256:(iy+1)*256,ix*256:(ix+1)*256][:,:,1] = 200
    print('x = %d, y = %d'%(image_x, image_y))    
    im.set_data(patch)
    plt.draw()                
    data.append((ix, iy))        
    return data

file_path = sys.argv[1]

## Missing a line
    print(dir(f))
    print(f.keys())
    dset = f['coords']
    length = len(dset)
    print(dset)
    coord_old = [0,0]
    for ncoord, coord in enumerate(dset):
        fig,ax = plt.subplots(figsize=(14,14))
        print(ncoord)
        if(ncoord>5): break
        wsi_object = WholeSlideImage(sys.argv[2])
        vis_level  = 0
        downsamples = wsi_object.wsi.level_downsamples[vis_level]
        patch_size = (256*4,256*4)
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size))#.convert("RGB"))
        coord_old = coord

        ## X Lines
        for i in range(4):
            ax.plot([0,1024],[256*i,256*i],'r--')
        ## Y Lines
        for j in range(4):
            ax.plot([256*j,256*j],[0,1024],'r--')
        im = ax.imshow(patch,origin='lower')
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        

print(data)
