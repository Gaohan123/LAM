# Custom transforms.


from PIL import Image

import torchvision.transforms

class Resize(torchvision.transforms.Resize):

    def __init__(self, size, interpolation='bilinear'):
        if interpolation == 'bilinear':
            interp_arg = Image.BILINEAR
        if interpolation == 'bicubic':
            interp_arg = Image.BICUBIC
        if interpolation == 'nearest':
            interp_arg = Image.NEAREST

        super().__init__(size=size, interpolation=interp_arg)
