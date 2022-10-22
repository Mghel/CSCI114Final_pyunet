import cv2
import numpy as np
import torch

def get_image(file, dim):
    img = cv2.resize(
        cv2.imread(
            file
        ),
        dim
    ) / 255

    return img

def get_mask(file, dim, num_classes=4):
    img = cv2.resize(
        cv2.imread(
            file,
            0
        ),
        dim
    )

    return img

def get_predicted_img(img, model, device='cpu', out_channels=4):
    model.to(device)

    input_img = img.transpose((2, 0, 1))

    x = torch.Tensor(np.array([input_img])).to(device)

    result = model.forward(x)
    result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.int32)
    result = result.transpose((1, 2, 0))

    return result