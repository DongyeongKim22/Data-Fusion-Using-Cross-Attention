import torch
import matplotlib.pyplot as plt
import numpy as np

def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def tensor_to_image(tensor):
    tensor = tensor.cpu().detach()
    tensor = tensor.numpy().transpose((1, 2, 0))  # CxHxW -> HxWxC
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def tensor_to_mask_image(tensor):
    tensor = tensor.cpu().detach()
    tensor = tensor.numpy().squeeze()  # CxHxW -> HxW (binary mask has only one channel)
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def visualize_results(source_imgs, target_imgs, outputs, instructions):
    num_samples = 7  # Number of samples to visualize
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        print(instructions[i])
        axes[i, 0].imshow(tensor_to_image(source_imgs[i]))
        axes[i, 0].set_title("Source Image")
        
        axes[i, 1].imshow(tensor_to_mask_image(outputs[i]), cmap='gray')
        axes[i, 1].set_title("Predicted Image")
        
        axes[i, 2].imshow(tensor_to_mask_image(target_imgs[i]), cmap='gray')
        axes[i, 2].set_title("Target Image")
        for ax in axes[i]:
            ax.axis('off')

    plt.show()