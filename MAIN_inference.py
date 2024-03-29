import argparse
import os
import json
import platform
import shutil
import time
from pathlib import Path
import warnings

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from numpy import array as npa
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib as mpl
# matplotlib setup
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['legend.framealpha'] = 0.65
mpl.rcParams['axes.labelsize'] = 'large'

from cycler import cycler  # used for changing default color sequence @ pyplot
# Close-up inset plots
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

# import module from parent of parent dir
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir_1 = os.path.dirname(current_dir)
parent_dir_2 = os.path.dirname(parent_dir_1)
sys.path.insert(0, parent_dir_2)
from utils_various import *


output_dir = '../../06 Output'
OUTPUT_TAG = ''

class Opt:
    agnostic_nms = False
    augment      = False
    classes      = None
    conf_thres   = 0.6
    device       = ''
    img_size     = 416
    iou_thres    = 0.5
    # output       = 'inference/output'
    save_txt     = False
    # source       = 'inference/images/img000002.jpg'
    update       = False
    view_img     = False
    weights      = ['runs/exp0/weights/best.pt']
    save_crop    = False  # i.e. save cropped RoI portion only
    show_crop    = False
    save_out     = False  # i.e. draw BB in original image and save


# Used in case of failure of the first SC localization attempt
# (this will be quite a rare event, after proper training)
crop_factor = 0.4  # --> center rectangle with sides @ 40% of original image
crop_xyxy = npa([(1 - crop_factor) / 2 * Camera.nu, (1 - crop_factor) / 2 * Camera.nv,
                 (.5 + crop_factor / 2) * Camera.nu, (.5 + crop_factor / 2) * Camera.nv], dtype=int)



def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5')

    parser.add_argument('--labels_dir',
                        help='Directory of the .json file containing the test labels',
                        type=str,
                        default='')
    parser.add_argument('--images_dir',
                        help='Directory containing the .jpg test images',
                        type=str,
                        default='')
    parser.add_argument('--output_tag',
                        help='Text label to append to inference .json output file',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    if args.labels_dir:
        TEST_LABELS_DIR = args.labels_dir
    else:
        # Default directory
        TEST_LABELS_DIR = '../../sharedData/test.json'

    if args.images_dir:
        TEST_IMAGES_DIR = args.images_dir
    else:
        # Default directory
        TEST_IMAGES_DIR = os.path.join(mySPEED_dir, 'images', 'test')

    if args.output_tag:
        OUTPUT_TAG = args.output_tag
    else:
        # Default empty tag
        OUTPUT_TAG = ''


    return TEST_LABELS_DIR, TEST_IMAGES_DIR, OUTPUT_TAG


if __name__ == '__main__':
    TEST_LABELS_DIR, TEST_IMAGES_DIR, OUTPUT_TAG = main()

# TEST_LABELS_DIR = '../../sharedData/test_rendering.json'
# TEST_IMAGES_DIR = '../../Rendering/rendezvous_render/frames/black_bg/noisy'

# Load JSON file with labels of original test set
with open(TEST_LABELS_DIR) as jFile:
    test_labels = json.load(jFile)


if OUTPUT_TAG == RENDERING_TAG:
    output_dir = os.path.join(output_dir, OUTPUT_TAG[1:])



class OriginalImage:

    """ Class for dataset inspection:
        provides access to individual images and corresponding ground truth labels.
    """

    def __init__(self, dir = 'inference/images/img000002.jpg'):
        self.dir = dir
        self.imgData = self.get_image()
        self.w = self.imgData.size[0]
        self.h = self.imgData.size[1]

    def get_image(self):
        """ Load image as PIL image. """

        img = Image.open(self.dir)#.convert('RGB')
        return img

    def get_cropped_ROI(self, xyxy):
        """ Crop ROI from original PIL image, given the: x_left, y_top, x_right, y_bottom
            coordinates measure from origin @ top-left corner.
        """
        x1, y1, x2, y2 = xyxy

        img_ROI = self.imgData.crop((x1,y1,x2,y2))  # (left, top, right, bottom)
        return img_ROI


def resize_crop(img_PIL):
    """
    If the largest dimension of the input PIL-image resulting from the crop (after
    identifying the RoI) is larger than the maximum allowed size (416 px), then
    resize it so that the following block of the pipeline (i.e. landmark regression)
    will have a reasonable computational burden.
    """
    # Look like it's worth to keep Antialiasing
    # (runtime increased at most by 4-5 ms only in worst cases)

    maxsize = (Opt.img_size, Opt.img_size)
    img_PIL.thumbnail(maxsize, Image.ANTIALIAS)
    return img_PIL

def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    box1: (x1, y1, x2, y2)
    box2: (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) intersection coordinates between box1, box2.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


##

def detect_ROI(source='inference/images/', Opt=Opt):
    view_img, save_txt, imgsz, save_crop, show_crop, save_out = \
        Opt.view_img, Opt.save_txt, Opt.img_size, Opt.save_crop, Opt.show_crop, Opt.save_out

    # N.B.
    # If source = (single img) instead of (folder of imgs)
    # then we will go backward from the parent directory of that file
    if os.path.isfile(source):
        source_folder = os.path.dirname(source)
    else:
        source_folder = source


    # Create folder for saving cropped RoIs, whenever requested
    if save_crop:
        roi_dir = os.path.join(Path(source_folder), 'roi')
        createDirectory(roi_dir)

    # Create folder for saving original image w/ BB overlay, whenever requested
    if save_out:
        out_dir = os.path.join(Path(source_folder).parent, 'output')
        createDirectory(out_dir)

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    xyxy_norm_matr = npa([])  # Bounding Box prediction
    probabilities = []        # Confidence score
    for path, img, im0, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # (W,H,C) --> (1,W,H,C)

        # Inference
        t0 = time.time()
        pred = model(img, augment=Opt.augment)[0]  # raw prediction (i.e. before NMS)

        # Apply NMS
        pred = non_max_suppression(pred, Opt.conf_thres, Opt.iou_thres, classes=Opt.classes, agnostic=Opt.agnostic_nms)

        detection_from_center_crop = False
        if pred[0] is None:  # i.e. FAILED detection
            detection_from_center_crop = True  # Need to flag this to properly recover prediction coords
            # When writing this comment: img008758, 008808, 012647
            # have been identified as critical, i.e. no BB detected

            # First of all we try analyzing the 40% center-crop rectangle,
            # then if needed we also try reducing the confidence threshold
            # that filters out all low confidence prediction, before applying NMS.
            # Threshold confidence is progressively reduced from the initial value
            # down to 0.4 (at most) until a detection is obtained.

            # Read image
            img = cv2.imread(path)  # BGR @ numpy:(H,W,C)
            img = img[crop_xyxy[1]:crop_xyxy[3],
                      crop_xyxy[0]:crop_xyxy[2],:]
            crop_after_fail_shape = img.shape  # this is needed for proper coordinate scaling

            # Padded resize
            img = letterbox(img, new_shape=Opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB @ numpy:(3x416x416)
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to(device)  # PIL --> numpy --> torch
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)  # (W,H,C) --> (1,W,H,C)

            # Inference
            pred = model(img, augment=Opt.augment)[0]  # raw prediction (i.e. before NMS)

            # NMS
            pred = non_max_suppression(pred, Opt.conf_thres, Opt.iou_thres, classes=Opt.classes,
                                       agnostic=Opt.agnostic_nms)

            lower_thresholds = np.arange(Opt.conf_thres-.1, .3, -.1)
            count = -1
            while (pred[0] is None) and count < len(lower_thresholds)-1:
                count += 1
                pred = model(img, augment=Opt.augment)[0]  # raw prediction (i.e. before NMS)
                pred = non_max_suppression(pred, lower_thresholds[count], Opt.iou_thres,
                                           classes=Opt.classes, agnostic=Opt.agnostic_nms)


            # If there is still no detection by YOLO, Landmark Regression Network
            # should in principle process the entire image, but given that such
            # a failure would expectedly occur only in the event of a very small
            # target, then we just "hope" the target is located inside the 30%
            # center rectangle of the frame. If we were to process the entire image,
            # it is very likely thar LRN would fail at detecting landmarks from a very
            # small target from an image that also gets downscaled to 260 x 416 pixels.
            if pred[0] is None:
                xyxy_norm = .5 + .5*Opt.img_size / npa([-Camera.nu ,-Camera.nv, +Camera.nu,+Camera.nv])
                probabilities.append(0)
                warnings.warn('No BB detected in %s, we just hope SC is inside xyxy-box: %s'
                              % (path, str(xyxy_norm)))
                if show_crop or save_crop:
                    img_ROI = OriginalImage(path).get_image()  # entire original image



        runtime = time.time() - t0



        # loop over detections
        for i, det in enumerate(pred):  # detections per image

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                if detection_from_center_crop:
                    ref_shape_scale = crop_after_fail_shape
                else:
                    ref_shape_scale = im0.shape
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ref_shape_scale).round()

                # Write results
                conf = 0.0 # initialize to zero in the event of no class detected
                for *xyxy, conf, cls in reversed(det):
                    # xywh_norm = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xyxy = torch.tensor(xyxy).tolist()
                    conf = torch.tensor(conf).tolist()

                    if detection_from_center_crop:  # if detection initially failed, but our previous zoom-in strategy
                                                    # was successful (i.e. pred != None), then we need to transform the
                                                    # coordinates of BB-prediction @ cropped to coordinates @ full-image
                        xyxy = (npa(xyxy) + npa([crop_xyxy[0], crop_xyxy[1], crop_xyxy[0], crop_xyxy[1]])).tolist()

                    if len(det) > 1:
                        print('%i DETECTIONS:' % len(det))
                        print(path, conf, xyxy)


                img_orig = OriginalImage(path)
                if show_crop or save_crop:
                    img_ROI = img_orig.get_cropped_ROI(xyxy)



                probabilities.append(conf)
                xyxy_norm = npa(xyxy) / npa([img_orig.w, img_orig.h, img_orig.w, img_orig.h])

        if Opt.save_out:
            label = '%.1f%%' % (100*conf)
            bgr_bb_color = [0, 215, 255]  # i.e. CSS 'gold' color
            plot_one_box(xyxy, im0, color=bgr_bb_color, label=label, line_thickness=3)
            img_name = os.path.basename(path)
            cv2.imwrite(os.path.join(out_dir,img_name), im0)


        # Stack BB predictions
        if len(xyxy_norm_matr) > 0:
            xyxy_norm_matr = np.vstack((xyxy_norm_matr, xyxy_norm))
        else:
            xyxy_norm_matr = xyxy_norm


        if show_crop:
            Image._show(img_ROI)

        if save_crop:
            img_name = os.path.basename(path)
            resize_crop(img_ROI).save(os.path.join(roi_dir, img_name))

    #   with open(txt_path + '.txt', 'a') as f:
    #       f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
    #
    #   if save_img or view_img:  # Add bbox to image
    #       label = '%s %.2f' % (names[int(cls)], conf)
    #       plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


    return xyxy_norm_matr, probabilities, runtime


## RUN INFERENCE

# Initialize inference
#set_logging()
device = select_device(Opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(Opt.weights, map_location=device)  # load FP32 model


## Inference on a few EXAMPLES
if False:
    img_dir = 'inference/images/'
    #img_dir = os.path.join(TEST_IMAGES_DIR)
    Opt.save_crop = True
    Opt.save_out = True
    detect_ROI(source=img_dir, Opt=Opt)


# INFERENCE MOSAIC
def inference_mosaic(img_list, img_per_row=3, filename='mosaic.jpg'):
    mosaic = []
    row = []
    count = 0

    for img_name in img_list:
        count += 1


        img = Image.open(os.path.join('inference/output', img_name))
        if len(row) == 0:
            row = np.array(img)
        else:
            row = np.hstack((row, np.array(img)))

        # whenever the no. images is not a multiple of row length,
        # just fill with black rectangle
        if count == len(img_list):  # and len(img_list) % img_per_row:
            while count % img_per_row:
                img = Image.new('RGB', (np.array(img).shape[1], np.array(img).shape[0]))
                row = np.hstack((row, np.array(img)))
                count += 1

        if not (count % img_per_row):
            if len(mosaic) == 0:
                mosaic = row
            else:
                mosaic = np.vstack((mosaic, row))
            row = []

    img_mosaic = Image.fromarray(mosaic)
    img_mosaic.save(os.path.join(output_dir,filename))


inference_image_list = os.listdir('inference/output/')


list_real_img = []
list_blackBG = []
list_earthBG = []

for filename in inference_image_list:

    if 'img' in filename:
        if 'real' in filename:
            list_real_img.append(filename)
        elif int( filename.lstrip('img').lstrip('0').rstrip('.jpg') ) < earthBGStart_ID:
            list_blackBG.append(filename)
        else:
            list_earthBG.append(filename)


inference_mosaic(list_real_img, 3, 'YOLO/yolo_inference_real.jpg')
inference_mosaic(list_blackBG, 3, 'YOLO/yolo_inference_blackBG.jpg')
inference_mosaic(list_earthBG, 3, 'YOLO/yolo_inference_earthBG.jpg')


## Inference on entire TEST set, to save and visualize cropped RoI
if False:
    img_dir = os.path.join(mySPEED_dir, 'images', 'test')
    Opt.save_crop = True
    Opt.save_out = False
    detect_ROI(source=img_dir, Opt=Opt)


## Inference on TEST to analyze performance

Opt.save_crop = False

# def get_idx_from_filename(filename, data):
#     count = 0
#     for item in data:
#         if item['filename'] == filename:
#             break
#         count += 1
#     return count

iou_test = []
prob_test = []
inference_data = []
for idx,img in enumerate(test_labels):

    # BB inference
    img_dir = os.path.join(TEST_IMAGES_DIR, img['filename'])


    xyxy_norm_inf, confidence, runtime = detect_ROI(source=img_dir, Opt=Opt)

    xyxy_inf = npa(xyxy_norm_inf * [Camera.nu, Camera.nv, Camera.nu, Camera.nv], dtype=int)
    inference_data.append({'image' : img['filename'],
                           'box' : xyxy_inf.tolist(),
                           'runtime' : runtime
                           })
    if not idx % 10:
        print('BB inference on %i/%i images' % (idx, len(test_labels)), end='\r')

    prob_test.append(confidence[0])


    try:
        # True BB label
        bb_true = img['bounding_box']
        TL = bb_true['TL']
        w = bb_true['w']
        h = bb_true['h']
        xyxy_norm_true = [TL[0], TL[1], TL[0]+w, TL[1]+h]

        # Compute IoU
        iou_test.append( iou(xyxy_norm_true, xyxy_norm_inf) )
    except Exception:
        pass


# save BB predictions (inference on whole test set)
with open('../../sharedData/' + 'yolov5_inference' + OUTPUT_TAG + '.json', 'w') as fp:
    json.dump(inference_data, fp)

# save IoU and confidence of individual test images
myDict = {
    'iou': iou_test,
    'probabilities': prob_test
}
with open('../../sharedData/' + 'yolov5_test_performance' + OUTPUT_TAG + '.json', 'w') as fp:
    json.dump(myDict, fp)



## LOAD INFERENCE RESULTS ON TEST DATA AND COMPUTE AP_50_95

# Load performance results on test set
with open('../../sharedData/yolov5_test_performance' + OUTPUT_TAG + '.json') as jFile:
    myDict = json.load(jFile)
iou_test = npa(myDict['iou']).squeeze()
prob_test = npa(myDict['probabilities']).squeeze()


def ap_at_iou(iou_vec, prob_vec, iou_min):

    # Sort examples according to prediction probability (ascending)
    sort_idx = np.flip(prob_vec.argsort())  # i.e. descending sort
    iou_vec, prob_vec = iou_vec[sort_idx], prob_vec[sort_idx]

    precision = []
    recall = []
    F1 = []
    is_iou_enough = (iou_vec >= iou_min)
    for score in prob_vec:
        TP = sum(np.logical_and((prob_vec >= score), is_iou_enough))
        # Alternatively:  sum((prob_test >= score) * (iou >= iou_min))
        # i.e. sum of a vector having entry = 1 only if True * True

        FP = sum(np.logical_and((prob_vec >= score), np.invert(is_iou_enough)))  # False Positives
        FN = sum(prob_vec < score)   # False Negatives

        P = TP / (TP + FP)
        R = TP / (TP + FN + 1e-16)

        precision.append(P)
        recall.append(R)
        F1.append( 2 * P * R / (P + R + 1e-16) )

    # Interpolated precision
    precision_interp = []
    for k in np.arange(len(precision)):
        precision_interp.append(max(precision[k:]))

    # Average precision computed as the integral of: P(R)
    AP = np.trapz(y=precision, x=recall)


    return AP, precision_interp, recall, F1



def ap_50_95(iou_vec, prob_vec):

    IoU_tresholds = np.arange(0.5,1.0,0.05)

    AP_50_95 = npa([])
    P_50_95 = np.empty( (0,len(prob_vec)), float)
    R_50_95 = np.empty( (0,len(prob_vec)), float)
    F1_50_95 = np.empty( (0,len(prob_vec)), float)

    for IoU in IoU_tresholds: # i.e. 10 steps 0.50, 0.55,... 0.95
        print('Computing AP @ IoU: %.2f' % IoU, end = '\r')
        AP, precision, recall, F1 = ap_at_iou(iou_vec, prob_vec, IoU)
        AP_50_95 = np.append(AP_50_95, AP)
        P_50_95  = np.vstack( (P_50_95, precision) )
        R_50_95  = np.vstack( (R_50_95, recall) )
        F1_50_95 = np.vstack( (F1_50_95, F1) )
    AP_50_95 = np.mean(AP_50_95)

    return AP_50_95, P_50_95, R_50_95, F1_50_95



# Compute performance metrics
IoU_mean = np.mean(iou_test)
IoU_median = np.median(iou_test)
AP_50_95, P_50_95, R_50_95, F1_50_95 = ap_50_95(iou_test, prob_test)
print('YOLOv5 PERFORMANCE ON TEST SET:')
print('AP_50-95 = %.4f\tIoU_mean = %.4f\tIoU_median = %.4f'
      % (AP_50_95, IoU_mean, IoU_median))


myDict['ap_50_95'] = AP_50_95.tolist()
myDict['p_50_95']  = P_50_95.tolist()
myDict['r_50_95']  = R_50_95.tolist()
myDict['f1_50_95'] = F1_50_95.tolist()

# Save performance parameters
with open('../../sharedData/' + 'yolov5_test_performance' + OUTPUT_TAG + '.json', 'w') as fp:
    json.dump(myDict, fp)




## Plot inference performance: P(R) curves


with open('../../sharedData/yolov5_test_performance' + OUTPUT_TAG + '.json') as jFile:
    test_labels = json.load(jFile)
AP_50_95, P_50_95, R_50_95, F1_50_95 = npa(test_labels['ap_50_95']), npa(test_labels['p_50_95']), npa(test_labels['r_50_95']), npa(test_labels['f1_50_95'])

plt.figure(figsize=(8.5,4.9))  # get current size: ax.figure.get_size_inches()
plt.title('Test set performance')
plt.xlabel('Recall')
plt.ylabel('Precision')
ax = plt.gca()
#ax.set_aspect('equal')

# Plot successive lines according to custom CMap color ordering
n_lines = P_50_95.shape[0]
cmap = rgb_cmap
color_list = cmap(np.linspace(0.1,0.9,n_lines))
#manual_color_list = (cycler(color=['r','b','m','g']))
ax.set_prop_cycle(cycler(color=color_list))

# Plot P(R) curves for all IoU = [0.5:.05:0.95]
IoU_tresholds = ['%.2f' % IoU for IoU in np.arange(0.5, 1.0, 0.05)]
plt.plot(R_50_95.T, P_50_95.T, linewidth=1)
ax.legend(IoU_tresholds, loc=4, title = 'IoU$_{\mathrm{min}}$')


# Close up of upper-right region (Inset Axes)
axins = zoomed_inset_axes(ax, zoom=6, loc=3) # zoom-factor: 6,   location: | 2  1 |
                                             # (e.g. 3 is lower-left)      | 3  4 |
axins.set_prop_cycle(cycler(color=color_list))
axins.plot(R_50_95.T, P_50_95.T, linewidth=1)
x1, x2, y1, y2 = 0.95, 1.004, 0.97, 1.003  # Crop coordinates
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
# connect the inset axes and the area in the parent axes
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.7", alpha=0.65)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='navajowhite', alpha=0.5)
ax.text(0.52, 0.86,
        'AP$_{50}^{95} = %.2f$ %%\nIoU$_\mathrm{mean} = %.2f$ %%\nIoU$_\mathrm{median} = %.2f$ %%'
        % (AP_50_95*100, IoU_mean*100, IoU_median*100),
        transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(os.path.join(output_dir, 'YOLO', 'prec_recall_50_95' + '.pdf'),
            bbox_inches='tight')


##
# Opt.show_crop = True
# xyxy_norm_inf = detect_ROI(source='/Users/massimopiazza/SPEED_MP/images/train/img012647.jpg', Opt=Opt)
# Opt.show_crop = False