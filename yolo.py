import colorsys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image
from osgeo import gdal
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input, stretch_16to8,
                         resize_image)
from utils.utils_bbox import DecodeBox

class YOLO(object):
    _defaults = {
        
        "model_path"        : 'model_data/0701mymodelv5_8.pth',
        "classes_path"      : 'model_data/rare_classes.txt',
        
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        
        "input_shape"       : [320, 320],
        
        "phi"               : 'l',
        
        "confidence"        : 0.3,
        
        "nms_iou"           : 0.5,
        
        "letterbox_image"   : False,
        
        "cuda"              : True,
        "VOCdevkit_path"    :"F:/Study/毕业论文/testgit/MyModelV3/VOCdevkit/VOC2007/JPEGImages/"
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    
    
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        
        self.colors = [(240,50,50),(50,255,255),(255, 255, 60)]
        self.generate()
    
    def generate(self):
        
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
    def atten_image(self, images, outpath):
      
      coarse = images[2][0]
      coarse  = coarse.cpu().numpy() * 100
      coarse  = np.squeeze(coarse)
      atten  = images[2][1]
      atten  = atten.cpu().numpy()
      atten  = np.squeeze(atten)
      
      tiff_driver = gdal.GetDriverByName("GTiff")
      filePath = r'F:\Study\temp1.tif'
      outImage = tiff_driver.Create(filePath,80,80,3,2)
      for i in range(3):
        outImage.GetRasterBand(i+1).WriteArray(coarse[i])
      outImage.FlushCache()
      pass
    
    def detect_image(self, image, imgName, crop = False):
        
        image_shape = np.shape(image)[1:3]
        
        image_data = np.expand_dims(preprocess_input(stretch_16to8(image)),0)
        # image_data = np.expand_dims(np.array(stretch_16to8(image),dtype=np.float32),0)
        
        imgPath = self.VOCdevkit_path + imgName+'.jpg'
        image = Image.open(imgPath)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                # images = images.type(torch.FloatTensor).cuda()
            
            outputs = self.net(images)
            
            outputs = np.array(outputs)[:,0]
            outputs = self.bbox_util.decode_box(outputs)
            
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 3))
        
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            
            
            del draw
        return image
    def get_FPS(self, image, test_interval):
        
        image_shape = np.shape(image)[1:3]
        image_data = np.expand_dims(preprocess_input(stretch_16to8(image)),0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            outputs = self.net(images)
            outputs = np.array(outputs)[:,0]
            outputs = self.bbox_util.decode_box(outputs)
            
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                
                outputs = self.net(images)
                outputs = np.array(outputs)[:,0]
                outputs = self.bbox_util.decode_box(outputs)
                
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
    def get_map_txt(self, image_id, image, class_names, map_out_path, map_vis, imageJpg, imageSavePath):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        
        image_shape = np.shape(image)[1:3]
        image_data = np.expand_dims(preprocess_input(stretch_16to8(image)),0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.type(torch.FloatTensor).cuda()
            
            outputs = self.net(images)
            outputs = np.array(outputs)[:,0]
            outputs = self.bbox_util.decode_box(outputs)
            
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
            if map_vis:
              imageJpg = self.drawingBox(results, imageJpg)
              imageJpg.save(imageSavePath)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
        f.close()
        return 
    def drawingBox(self, results, image):
      
      font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
      thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 3))
      top_label   = np.array(results[0][:, 6], dtype = 'int32')
      top_conf    = results[0][:, 4] * results[0][:, 5]
      top_boxes   = results[0][:, :4]
      for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            
            
            del draw
      return image