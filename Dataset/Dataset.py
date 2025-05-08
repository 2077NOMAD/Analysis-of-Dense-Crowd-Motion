from torch.utils.data import Dataset
import torch
import os
from transformers import AutoTokenizer
import cv2
import numpy as np

class Dataset(Dataset):
    def __init__(self, video_path,
                 class_to_idx,
                 subset,
                 device,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,):
        self.device = device
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.class_to_idx = class_to_idx
        self.samples = []
        self.cache = {}
        self.class_to_idx = class_to_idx
        self.make_dataset(
            video_path=video_path,
            subset=subset,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        data_dict = torch.load(self.samples[idx][0], weights_only=True)
        label = self.samples[idx][1]
        video_tensor = data_dict['Video']
        llm_tensor = data_dict['LLM']
        video_img = video_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
        video_img = cv2.cvtColor((video_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        decoded_text = self.tokenizer.decode(llm_tensor.squeeze(), skip_special_tokens=True)
        result = ({'Video': video_img, 'LLM': decoded_text}, label)
        self.cache[idx] = result
        return result


    def make_dataset(self,video_path, subset):
        subset_dir = os.path.join(video_path, subset)
        for class_dir in os.listdir(subset_dir):
            class_path = os.path.join(subset_dir, class_dir)
            if os.path.isdir(class_path) and class_dir in self.class_to_idx:
                for file in os.listdir(class_path):
                    if file.endswith('.pt'):
                        file_path = os.path.join(class_path, file)
                        label = self.class_to_idx[class_dir]
                        self.samples.append((file_path, label))
