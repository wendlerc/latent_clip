from timm.data import create_transform
from torchvision import transforms

def create_timm_transform(image_size, 
                          is_training=True, 
                          hflip=0.0, 
                          scale=(0.66, 1), 
                          color_jitter=(0.2,0.2,0.2,0.02), 
                          interpolation='random'):
    # scale_lower, scale_upper = scale
    # brightness, contrast, saturation, hue = color_jitter
    if isinstance(image_size, (tuple, list)):
        assert len(image_size) >= 2
        input_size = (3,) + image_size[-2:]
    else:
        input_size = (3, image_size, image_size)

    primary_tfl, secondary_tfl, final_tfl = create_transform(
                input_size=input_size,
                is_training=is_training,
                hflip=hflip,
                re_mode='pixel',
                separate=True,
                scale=scale,
                color_jitter=color_jitter,
                interpolation=interpolation,
    )
    # compose only primary and secondary transforms
    # finals transform would convert to tensor and normalize, we perfer to deal with PIL.Image
    transform = transforms.Compose([primary_tfl, secondary_tfl])
    return transform