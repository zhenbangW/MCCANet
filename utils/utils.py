import numpy as np
from PIL import Image
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def preprocess_input(image):
    image /= 255.0
    return image
# def stretch_16to8(img, lower_percent=2, higher_percent=98):
#     band_count = img.shape[0]
#     in_h       = img.shape[1]
#     in_w       = img.shape[2]

#     img_array = img.reshape(band_count, -1)
#     # min = np.min(bands, axis=1)
#     # max = np.max(bands, axis=1)

#     mean = np.mean(img_array, axis=1).reshape(band_count,-1)
#     std = np.std(img_array, axis=1, ddof=1).reshape(band_count,-1)

#     # max_min = np.repeat((max - min).reshape(4,-1), 320*320, axis=1).reshape(4,320,320)


#     d = np.repeat(mean, in_h*in_w, axis=1).reshape(band_count,in_h,in_w)
#     e = np.repeat(std, in_h*in_w, axis=1).reshape(band_count,in_h,in_w)
#     out = (img - d)/e
    
#     return out

def stretch_16to8(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands, dtype=np.float32)
    n = bands.shape[0]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)
        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out

def get_classNums(num_classes, sample_path):
   
    classes = [x for x in range(num_classes)]
    classNums = [0 for x in range(num_classes)]
    with open(sample_path,encoding= 'utf-8') as f:
      lines = f.readlines()
      for line in lines:
        sample = line.split()[1:]
        for num in sample:
          classtype = num.split(',')[-1]
          classNums[int(classtype)]+=1
    return classNums