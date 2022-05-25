import numpy as np
import torch
from .models import _legacy_timm_model
import PIL.Image as pil_img


def correct_img_rotation(
    model: _legacy_timm_model.TimmModule, img: np.ndarray
) -> np.ndarray:

    model = model.eval()

    img_transform = model.get_transform()
    img_tns = img_transform(pil_img.fromarray(img))
    model_input = img_tns.reshape(1, *img_tns.shape).to(model.device)

    with torch.no_grad():
        pred = torch.argmax(model.forward(model_input)[0, ...]).item()

    if pred == 0:
        return img

    result = np.rot90(img, k=4 - pred)

    return result
