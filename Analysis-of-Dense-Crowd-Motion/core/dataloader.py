from torch.utils.data import DataLoader
from Dataset.Dataset import Dataset

def get_data(opt, subset, transforms):
    spatial_transform, temporal_transform, target_transform = transforms
    return Dataset(opt.video_path,
                      opt.class_to_idx,
                      subset,
                      opt.device,
                      spatial_transform,
                      temporal_transform,
                      target_transform)


def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_data(opt, 'train', transforms)

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_data(opt, 'validation', transforms)

def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_data(opt, 'test', transforms)


def get_data_loader(opt, dataset, shuffle, batch_size=0, num_workers=None):
    batch_size = opt.batch_size if batch_size == 0 else batch_size
    workers = opt.n_threads if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=opt.dl
    )


