from typing import Union, List
from functools import partial
import torch
import torch.nn as nn


def _listify(x: Union[int, List[int]]):
    return [x] if isinstance(x, int) else x


def _extend_to(x: List, n: int):
    if n > 1 and len(x) == 1:
        x = x + (x * (n - 1))
    return x


def _to_aligned_lists(*args: Union[int, List[int]]):
    """Extends all arguments to lists of the same length"""
    args = list(map(_listify, args))
    n = max(map(len, args))
    if n == 1:
        return args
    args = list(map(partial(_extend_to, n=n), args))
    min_len = min(map(len, args))
    if min_len != n:
        raise ValueError("Cannot align arguments.\nArguments must be int, list of length 1 or list of equal length.")
    return args


def _output_shape_(width, height, channel, kernel, kernel_stride, padding, save_list):
    save_list.append((channel[0], width, height))
    if len(channel) == 1:
        return save_list
    width = ((width + 2 * padding[0] - kernel[0]) // kernel_stride[0]) + 1
    height = ((height + 2 * padding[0] - kernel[0]) // kernel_stride[0]) + 1
    return _output_shape_(width, height, channel[1:], kernel[1:], kernel_stride[1:], padding[1:], save_list)


def _output_shape(width, height, channel, kernel, kernel_stride, padding, return_sequence=False):
    output_shape = _output_shape_(width, height, channel, kernel, kernel_stride, padding, [])
    if return_sequence:
        return output_shape
    return output_shape[-1]


class ImageEncoder(nn.Module):
    def __init__(self, in_channel, width, height, out_channel, kernel, kernel_stride, padding, dense):
        super(ImageEncoder, self).__init__()
        channels = [in_channel] + out_channel
        cnn_layers = []
        for c_in, c_out, k, k_s, p in zip(channels, out_channel, kernel, kernel_stride, padding):
            cnn_layers.append(nn.Conv2d(c_in, c_out, k, k_s, p))
            cnn_layers.append(nn.ReLU())
        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.flatten = nn.Flatten(start_dim=1)
        c_out, w_out, h_out = _output_shape(width, height, channels, kernel, kernel_stride, padding, False)
        dense_layers = []
        for d_in, d_out in zip([c_out * w_out * h_out] + dense, dense):
            dense_layers.append(nn.Linear(d_in, d_out))
            dense_layers.append(nn.ReLU())
        self.dense_layers = nn.Sequential(*dense_layers[:-1])

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, output_size, kernel, kernel_stride, padding, dense):
        super(ImageDecoder, self).__init__()
        self.output_size = output_size[1:]
        c_out, w_out, h_out = output_size[0]
        dense_layers = []
        for d_in, d_out in zip(dense, dense[1:] + [c_out * w_out * h_out]):
            dense_layers.append(nn.Linear(d_in, d_out))
            dense_layers.append(nn.ReLU())
        self.dense_layers = nn.Sequential(*dense_layers)
        self.unflatten = nn.Unflatten(1, (c_out, w_out, h_out))
        channels = [c for c, _, _ in output_size]
        self.conv_layers, self.relu, module = [], [], []
        for c_in, c_out, k, k_s, p in zip(channels, channels[1:], kernel, kernel_stride, padding):
            self.conv_layers.append(nn.ConvTranspose2d(c_in, c_out, k, k_s, p))
            self.relu.append(nn.ReLU())
            module.append(self.conv_layers[-1])
            module.append(self.relu[-1])
        self.cnn_layers = nn.Sequential(*module[:-1])
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_layers(x)
        x = self.unflatten(x)
        for conv, relu, output_size in zip(self.conv_layers, self.relu[:-1], self.output_size):
            x = conv(x, output_size=(x.shape[0], *output_size))
            x = relu(x)
        x = self.conv_layers[-1](x, output_size=(x.shape[0], *self.output_size[-1]))
        x = self.sig(x)
        return x


class ImageAutoencoder(nn.Module):
    def __init__(self, in_channel, width, height, out_channel, kernel, kernel_stride, padding, dense):
        super(ImageAutoencoder, self).__init__()
        out_channel, kernel, kernel_stride, padding = _to_aligned_lists(out_channel, kernel, kernel_stride, padding)
        self.image_encoder = ImageEncoder(in_channel, width, height, out_channel, kernel, kernel_stride, padding, dense)
        self.output_size = _output_shape(width, height, [in_channel] + out_channel, kernel, kernel_stride, padding,
                                         True)
        self.image_decoder = ImageDecoder(self.output_size[::-1], kernel[::-1], kernel_stride[::-1], padding[::-1],
                                          dense[::-1])

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.image_decoder(x)
        return x


if __name__ == "__main__":
    # print(_output_shape(32, 32, [1, 8, 16], [3,, [2, 2], [0, 0], return_sequence=True))
    # encoder = ImageEncoder(1, 32, 32, [8, 16], [3, 3], [2, 2], [0, 0], [254, 128, 32])
    # print(encoder)
    batch = 10
    channel = 17
    width, heigth = 32, 32
    foo = torch.rand((batch, channel, width, heigth))
    autoencoder = ImageAutoencoder(channel, width, heigth, [8, 16], 3, 2, 5, [254, 128, 32])
    print(autoencoder)
    print(autoencoder(foo).shape)
