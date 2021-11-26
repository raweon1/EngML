import torch
import matplotlib.pyplot as plt
import numpy as np


def plot(model, img, figsize=(20, 10), label=""):
    with torch.no_grad():
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.suptitle(f"Label: {label}")
        ax[0].imshow(img.squeeze().numpy(), cmap="gray")
        ax[0].set_title("Orig")
        ax[1].imshow(model(img.unsqueeze(0)).squeeze().numpy(), cmap="gray")
        ax[1].set_title("Recon")
        return fig


def gauss_filter(img_tensor, mu, sigma):
    normal = torch.distributions.Normal(mu, sigma)
    return img_tensor + normal.sample(img_tensor.shape)


def remove_pixel_filter(img_tensor, zero_chance, fill_chance, fill_value):
    img_tensor = img_tensor.clone()
    uniform = torch.distributions.Uniform(0, 1)
    index = uniform.sample(img_tensor.shape)
    img_tensor[index < zero_chance] = 0
    img_tensor[index > 1 - fill_chance] = fill_value
    return img_tensor


def change_n_pixel_filter(img_tensor: torch.Tensor, n):
    shape = img_tensor.shape
    img_tensor = img_tensor.clone().flatten(start_dim=1)
    for i in range(img_tensor.shape[0]):
        index = np.random.choice(img_tensor.shape[1], n, replace=False)
        mask = img_tensor[i, index] >= 0.5
        img_tensor[i, index[mask]] = 0
        img_tensor[i, index[~mask]] = 1
    return img_tensor.reshape(shape)
