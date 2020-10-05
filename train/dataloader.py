import cv2
import numpy as np
import torch
import torchvision
import opencv_transforms.functional as FF
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random
from torchvision import datasets
from PIL import Image
    
class GetImageFolder(datasets.ImageFolder):   
    """
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        sketch_net: The network to convert color image to sketch image
        ncluster: Number of clusters when extracting color palette.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    
     Getitem:
        img_edge: Edge image
        img: Color Image
        color_palette: Extracted color paltette
    """
    def __init__(self, root, transform, is_pair=False, start='Gray', end='RGB', hint='Scribble', sketch_net=None):
        super(GetImageFolder, self).__init__(root, transform)
        self.is_pair = is_pair
        self.start = start
        self.end = end
        self.hint = hint
        self.sketch_net = sketch_net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        if self.is_pair:
            img = img[:, 0:512, :]
        img = self.transform(img)
        img_color = img
        
        if self.start == 'Gray' or self.end == 'Gray':
            img_gray = FF.to_grayscale(img_color, num_output_channels=1)
        
        if self.start == 'Edge' or self.end == 'Edge':
            if self.sketch_net:
                with torch.no_grad():
                    img_temp = make_tensor(img_color)
                    img_edge = self.sketch_net(img_temp.unsqueeze(0).to(self.device))
                    img_edge = img_edge.squeeze().cpu().numpy()
                img_edge = img_edge * 255
                img_edge = img_edge.astype(np.uint8)[:, :, np.newaxis]
            
        if self.start == 'XDoG' or self.end == 'XDoG':
            img_XDoG = XDoG(img)      
            
        if self.start in {'L', 'ab', 'Lab'} or self.end in {'L', 'ab', 'Lab'}:
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)    
            img_l = img_lab[:, :, 0:1]
            img_ab = img_lab[:, :, 1:3]
                
        if self.start == 'RGB':
            img_start = img_color
        elif self.start == 'Lab':
            img_start = img_lab
        elif self.start == 'L':
            img_start = img_l   
        elif self.start == 'ab':
            img_start = img_ab
        elif self.start == 'Gray':
            img_start = img_gray
        elif self.start == 'Edge':
            img_start = img_edge
        elif self.start == 'XDoG':
            img_start = img_XDoG     
        elif self.start == 'Clustered':
            img_start = img_clustered
            
        if self.end == 'RGB':
            img_end = img_color
        elif self.end == 'Lab':
            img_end = img_lab
        elif self.end == 'L':
            img_end = img_l  
        elif self.end == 'ab':
            img_end = img_ab
        elif self.end == 'Gray':
            img_end = img_gray
        elif self.end == 'Edge':
            img_end = img_edge
        elif self.end == 'XDoG':
            img_end = img_XDoG  
        elif self.end == 'Clustered':
            img_end = img_clustered
            
        if self.hint is not None:
            small_img = cv2.resize(img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
            img_clustered = clustering(small_img, (7,10))   
            img_clustered = cv2.resize(img_clustered, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
            
            if self.hint == 'Scribbles':
                mask = get_mask(img_color, self.hint, num_point_range=(30, 50), length_range=(100, 150), thickness_range=(3, 7))
            elif self.hint == 'Points':
                mask = get_mask(img_color, self.hint, num_point_range=(75, 150), length_range=None, thickness_range=(3, 7))
                
            img_hint = (1-mask)*img_start + mask*img_clustered

        else:
            img_hint = img_color
       

        img_start = make_tensor(img_start)   
        img_end = make_tensor(img_end)
        img_hint = make_tensor(img_hint)

        return img_start, img_end, img_hint
    
def make_tensor(img):
    img = FF.to_tensor(img)
    return img
    
def show_example(img_list, size):
    plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0, wspace=0)
    num_image = len(img_list)
    
    for i in range(1, num_image+1):
        ax = plt.subplot(1, num_image, i)
        img = img_list[i-1]
        result = torch.cat([img],dim=-1)
        plt.imshow(np.transpose(vutils.make_grid(result, nrow=1, padding=5, normalize=True).cpu(),(1,2,0)), aspect='auto')
        plt.axis("off")

    plt.show()
    
def convert_color(img, mode):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if len(img.shape) == 3: # channel x width x height
        img = img[np.newaxis, :, :, :] # batchsize(1) x channel x width x height
        
    img = np.transpose(img, (0, 2, 3, 1))  # batchsize x width x height x channel
    converted_img_list = []
    batch_size = img.shape[0]
    
    for i in range(batch_size):
        one_img = img[i, :, :, :]
        one_img = one_img * 255
        one_img = one_img.astype(np.uint8)
        if mode=='to_RGB':
            converted_img = cv2.cvtColor(one_img, cv2.COLOR_LAB2RGB)
        elif mode=='to_Lab':
            converted_img = cv2.cvtColor(one_img, cv2.COLOR_RGB2LAB) 
        converted_img = FF.to_tensor(converted_img)
        converted_img_list.append(converted_img)

    converted_tensor = torch.stack(converted_img_list, dim=0)

    return converted_tensor

def concat_lab(img_l, img_ab):
    if len(img_l.size()) == 3:
        img_l = img_l.unsqueeze(1)
    img_lab = torch.cat([img_l, img_ab], dim=1)
    return img_lab

def XDoG(img, sigma = 0.9, k = 3.0, t = 0.998, e = -0.1, p = 30):
    img = img.astype(np.float32)/255

    Ig1 = cv2.GaussianBlur(img, (3, 3), sigma, sigma)
    Ig2 = cv2.GaussianBlur(img, (3, 3), sigma * k, sigma * k)

    Dg = (Ig1 - t * Ig2)

    Dg[Dg<e] = 1
    Dg[Dg>=e]= 1 + np.tanh(p * Dg[Dg>=e])

    Dg[Dg>1.0] = 1.0
    Dg = Dg * 255

    return Dg.astype(np.uint8)

def get_mask(image, mode, num_point_range, length_range, thickness_range):
    width = image.shape[0]
    height = image.shape[1]
    mask = np.zeros((width, height, 1), dtype=np.uint8)
    num_point = random.randint(*num_point_range)

    for _ in range(num_point):
        if mode == 'Scribbles':
            mask = draw_scribbles(image, mask, length_range, thickness_range)
        elif mode == 'Points':
            mask = draw_points(mask, thickness_range)
        
    return mask

def draw_scribbles(image, mask, length_range, thickness_range):
    width, height, _ = mask.shape
    length = random.randint(*length_range)
    thickness = random.randint(*thickness_range)
    point_x = random.randint(0, width-1-thickness)
    point_y = random.randint(0, height-1-thickness)
    direction = random.randint(0, 9)

    for _ in range(length):
        noise = random.randint(-1, 2)
        if direction in {0, 4}:
            point_x = point_x
            point_y += noise
        elif direction in {1, 2, 3}: 
            point_x += 1
            point_y += noise                
        elif direction in {5, 6, 7}:
            point_x -= 1
            point_y += noise     
        elif direction in {8}:
            point_y += noise     
            
        noise = random.randint(-1, 2)
        if direction in {2, 6}:
            point_y = point_y
            point_x += noise
        elif direction in {7, 0, 1}: 
            point_y += 1
            point_x += noise
        elif direction in {3, 4, 5}:
            point_y -= 1
            point_x += noise
        elif direction in {8}:
            point_x += noise     
            
        mask[point_x:point_x+thickness, point_y:point_y+thickness] = 1

    return mask

def draw_points(mask, thickness_range):
    width, height, _ = mask.shape
    thickness = random.randint(*thickness_range)
    point_x = random.randint(0, width-1-thickness)
    point_y = random.randint(0, height-1-thickness)
    mask[point_x:point_x+thickness, point_y:point_y+thickness] = 1
    
    return mask

def clustering(image, K_range):
    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    K = random.randint(*K_range)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    
    return res2
    