from PIL import Image
import numpy as np
import torch


def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def batch_tensor_to_pil(img_tensor):
    # Convert tensor of shape [batch_size, channels, height, width] to a list of PIL Images
    return [tensor_to_pil(img_tensor, i) for i in range(img_tensor.shape[0])]


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def batched_pil_to_tensor(images):
    # Takes a list of PIL images and returns a tensor of shape [batch_size, height, width, channels]
    return torch.cat([pil_to_tensor(image) for image in images], dim=0)
