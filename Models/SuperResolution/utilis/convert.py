from PIL import Image
from torchvision import transforms


def convert_ycbcr_to_rgb(y, cb, cr):
    y = transforms.ToPILImage()(y)
    cb = Image.fromarray(cb.numpy(), mode='L')
    cr = Image.fromarray(cr.numpy(), mode='L')

    rgb = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')

    return rgb
