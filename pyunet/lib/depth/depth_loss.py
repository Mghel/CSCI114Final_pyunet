import torch
import torch.nn as nn
import pytorch_msssim
import pywt

def wavelet_loss(predictions, depthmap, wavelet_level=1, weight_wavelet=0.2):
    # Calculate wavelet transform of predictions and depthmap
    wavelet_predictions = pywt.wavedec2(predictions.cpu().detach().numpy(), 'db1', level=wavelet_level)
    wavelet_depthmap = pywt.wavedec2(depthmap.cpu().detach().numpy(), 'db1', level=wavelet_level)

    # Compute the difference in wavelet coefficients
    wavelet_diff = 0.0
    for coef_pred, coef_depth in zip(wavelet_predictions, wavelet_depthmap):
        wavelet_diff += torch.mean(torch.abs(torch.tensor(coef_pred) - torch.tensor(coef_depth)))

    # Weighted wavelet loss
    weighted_wavelet_loss = weight_wavelet * wavelet_diff

    return weighted_wavelet_loss

def depth_loss(predictions, depthmap, weight_pointwise=0.4, weight_gradient=0.2, weight_ssim=0.2):
    mse_loss = nn.MSELoss()

    # Compute pointwise loss
    loss_pointwise = mse_loss(predictions, depthmap)

    # Gradient loss
    gradient_kernel_x = torch.tensor([[0, 0, 0],
                                      [-1, 0, 1],
                                      [0, 0, 0]], dtype=torch.float32).to(predictions.device).view(1, 1, 3, 3)

    gradient_kernel_y = torch.tensor([[0, -1, 0],
                                      [0, 0, 0],
                                      [0, 1, 0]], dtype=torch.float32).to(predictions.device).view(1, 1, 3, 3)

    gradient_predictions_x = nn.functional.conv2d(predictions, gradient_kernel_x, padding=1)
    gradient_depthmap_x = nn.functional.conv2d(depthmap, gradient_kernel_x, padding=1)

    gradient_predictions_y = nn.functional.conv2d(predictions, gradient_kernel_y, padding=1)
    gradient_depthmap_y = nn.functional.conv2d(depthmap, gradient_kernel_y, padding=1)

    squared_diff_x = (gradient_predictions_x - gradient_depthmap_x) ** 2
    squared_diff_y = (gradient_predictions_y - gradient_depthmap_y) ** 2

    gradient_loss_x = torch.mean(squared_diff_x)
    gradient_loss_y = torch.mean(squared_diff_y)

    g_loss = gradient_loss_x + gradient_loss_y

    # SSIM loss
    ssim_loss = 1.0 - pytorch_msssim.ssim(predictions, depthmap)

    # Wavelet loss
    wavelet_loss_value = wavelet_loss(predictions, depthmap)

    # Combine all the losses
    total_loss = (
        weight_pointwise * loss_pointwise +
        weight_gradient * g_loss +
        weight_ssim * ssim_loss +
        wavelet_loss_value
    )

    return total_loss
