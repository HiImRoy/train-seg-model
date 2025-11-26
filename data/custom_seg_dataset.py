# -*- coding: utf-8 -*-
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomSegDataset(BaseDataset):
    """
    A custom dataset class for segmentation that handles the following structure:
    - dataroot/
      - train_img/
      - train_lab/
      - test_img/ (used for validation)
      - test_lab/ (used for validation)
    
    It also handles mixed label extensions (.png, .jpg).
    """
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # Determine directories based on phase (train or val)
        phase = opt.phase
        if phase == 'val':
            self.dir_img = os.path.join(opt.dataroot, 'test_img')
            self.dir_lab = os.path.join(opt.dataroot, 'test_lab')
        else: # 'train'
            self.dir_img = os.path.join(opt.dataroot, 'train_img')
            self.dir_lab = os.path.join(opt.dataroot, 'train_lab')

        self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))
        
        self.lab_paths = []
        print(f"Loading labels for {len(self.img_paths)} images from {self.dir_lab}...")
        for img_path in self.img_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Check for .png first, then .jpg
            lab_path_png = os.path.join(self.dir_lab, base_name + '.png')
            lab_path_jpg = os.path.join(self.dir_lab, base_name + '.jpg')
            
            if os.path.exists(lab_path_png):
                self.lab_paths.append(lab_path_png)
            elif os.path.exists(lab_path_jpg):
                self.lab_paths.append(lab_path_jpg)
            else:
                print(f"Warning: No label found for image {img_path}")
        
        assert len(self.img_paths) == len(self.lab_paths), \
            "The number of images and labels do not match. Please check your dataset."

        # --- FIX: Create separate transforms for image and mask ---
        # Transform for the input image (includes normalization)
        self.transform_img = get_transform(opt)
        
        # A simpler transform for the label mask (only resize, crop, and convert to tensor)
        # We get the basic parameters from the full transform options
        transform_list = []
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.NEAREST))
        if 'crop' in opt.preprocess:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        
        transform_list.append(transforms.ToTensor())
        self.transform_lab = transforms.Compose(transform_list)


    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        img_path = self.img_paths[index]
        lab_path = self.lab_paths[index]
        
        A = Image.open(img_path).convert('RGB')
        B = Image.open(lab_path).convert('L') # Convert label to grayscale

        # --- FIX: Apply the correct transform to each ---
        A = self.transform_img(A)
        B = self.transform_lab(B)

        return {'image': A, 'label': B, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
