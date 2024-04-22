"""Wraps the AndroidEnv environment"""
import os
import traceback
from PIL import Image

def convert_torch_image_to_pillow(tensor_img):
    if len(tensor_img.shape) == 4:
        assert tensor_img.shape[0] == 1
        tensor_img = tensor_img.squeeze(0)
    
    tensor_img = tensor_img.permute(1, 2, 0)

    np_img = tensor_img.detach().cpu().numpy()
    
    return Image.fromarray(np_img, mode="RGB")


def convert_np_image_to_pillow(np_img):
    return Image.fromarray(np_img, mode="RGB")


def save_pillow_images_to_gif(images, filename="tmp"):
    try:
        images[0].save(f"{filename}.gif", 
                       save_all=True, 
                       append_images=images[1:], 
                       duration=500, 
                       loop=0)
        return True
    except:
        traceback.print_exc()
        return False