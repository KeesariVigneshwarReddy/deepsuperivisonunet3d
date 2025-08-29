import zarr, pandas as pd, json, os, cv2, numpy as np, random
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

class CustomDataset(Dataset):
    
    def __init__(self, 
                 base_dir,
                 run_ids,
                 versions,
                 classes,
                 voxel_spacing=10.012444196428572,
                 multiplier=256):
        
        self.base_dir = base_dir
        self.run_ids = run_ids
        self.versions = versions
        self.classes = classes
        self.voxel_spacing = voxel_spacing
        self.multiplier = multiplier
        self.images_masks = []

        print('Gathering ground truth coordinates of protein complexes...')
        self.coordinates = pd.DataFrame(columns=['run_id', 'class', 'x', 'y', 'z'])
        for run_id in tqdm(self.run_ids):
            for class_ in self.classes:
                filepath = self.base_dir.replace('static', 'overlay') + '/' + run_id + '/' + 'Picks' + '/' + class_ + '.json'
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for point in data['points']:
                        location = point['location']
                        x, y, z = location['x'] / self.voxel_spacing, location['y'] / self.voxel_spacing, location['z'] / self.voxel_spacing
                        self.coordinates.loc[len(self.coordinates)] = [run_id, class_, round(x), round(y), round(z)]
        print('Dataframe shape =', self.coordinates.shape)
        print(self.coordinates.head())
        print('......')
        print('......')
        
        print('Add a binary channel to the image and preparing ground truth segmentation masks...')
        for run_id in tqdm(self.run_ids):
            mask = self.create_mask(self.coordinates[self.coordinates['run_id']==run_id])
            for version in self.versions:
                image_path = base_dir + '/' + run_id + '/' + 'VoxelSpacing10.000' + '/' + version + '/' + str(0)
                image = zarr.open(image_path)[:]
                image = (image - image.min()) / (image.max() - image.min())
                image = self.add_binary_channel(image)
                self.images_masks.append((image, mask))
        print('Total (image, mask) pairs =', len(self.images_masks))
        print('Shape =', self.images_masks[0][0].shape, self.images_masks[0][1].shape)
        print('Use visualize function by passing the idx...')

    def __len__(self):
        return self.multiplier * len(self.images_masks)

    def __getitem__(self, idx):
        image, mask = self.images_masks[idx // self.multiplier]
        image, mask = self.augment(image, mask)
        image = torch.from_numpy(image.copy()).float()
        mask  = torch.from_numpy(mask.copy()).long()
        return image, mask
        
    def create_mask(self, coordinates):
        mask = np.zeros((7, 184, 630, 630), dtype=np.uint8)
        
        for _, row in coordinates.iterrows():
            c, z, y, x = self.classes.index(row['class']) + 1, row["z"], row['y'], row['x']
            mask[c, z, y, x] = 1
        
        mask[0] = 1 - np.clip(mask[1:].sum(axis=0), 0, 1)
        return mask
        
    def add_binary_channel(self, image):
        binary_image = np.zeros_like(image)
        for z in range(image.shape[0]):
            slice_z = (image[z] * 255).astype(np.uint8)
            _, binary_slice = cv2.threshold(slice_z, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_slice = 255 - binary_slice
            binary_slice = cv2.medianBlur(binary_slice, 7)
            binary_slice = binary_slice / 255
            binary_image[z] = binary_slice
        return np.stack([image, binary_image], axis=0)

    def visualize(self, idx):

        volume = self.images_masks[idx][0]
        num_channels, depth, height, width = volume.shape
        
        if num_channels != 2:
            raise ValueError("Volume must have 2 channels.")
        
        for z in range(depth):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            axs[0].imshow(volume[0, z], cmap='gray')
            axs[0].set_title(f"Channel 0 - Slice {z}")
            axs[0].axis('off')
            
            axs[1].imshow(volume[1, z], cmap='gray')
            axs[1].set_title(f"Channel 1 - Slice {z}")
            axs[1].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def random_crop_3d(self, image, mask, crop_shape=(96, 96, 96)):
        _, D, H, W = image.shape
        cd, ch, cw = crop_shape
        d0 = random.randint(0, D - cd)
        h0 = random.randint(0, H - ch)
        w0 = random.randint(0, W - cw)
        img_crop = image[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
        mask_crop = mask[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
        return img_crop, mask_crop
    
    def random_flip(self, image, mask, p=0.5):
        # Flip along z-axis (axis=1)
        if random.random() < p:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        # Flip along y-axis (axis=2)
        if random.random() < p:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=2)
        # Flip along x-axis (axis=3)
        if random.random() < p:
            image = np.flip(image, axis=3)
            mask = np.flip(mask, axis=3)
        return image, mask
    
    def random_rotate_90(self, image, mask):
        axis_pairs = [(1,2), (1,3), (2,3)]  # (D,H), (D,W), (H,W)
        axes = random.choice(axis_pairs)
        k = random.randint(0, 3)
        image = np.rot90(image, k=k, axes=axes)
        mask = np.rot90(mask, k=k, axes=axes)
        return image, mask
    
    def augment(self, image, mask):
        image, mask = self.random_crop_3d(image, mask, crop_shape=(96, 96, 96))
        image, mask = self.random_flip(image, mask, p=0.5)
        image, mask = self.random_rotate_90(image, mask)
        return image, mask