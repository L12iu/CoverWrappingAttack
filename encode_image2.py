import glob
import os

import bchlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from model import StegaStampEncoder

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        default='/opt/data/jiang/cross/CRoSS/image_jiang/gt2.png')
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/')
    parser.add_argument('--secret', type=str, default='Wooden')
    parser.add_argument('--secret_size', type=int, default=100)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    device = torch.device('cuda:0')
    model_path = '/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth'
    encoder = StegaStampEncoder(args)
    model_pth = torch.load(model_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(model_pth['model_encoder'])
    encoder.eval()
    encoder.to(device)

    width, height = 400, 400  # 保持 400x400 以匹配 encoder

    orisecret = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                           0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                           0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                           0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                           0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                           1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                           0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                           0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                           0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                           1, 0, 1, 0, 0, 0, 0, 0, 1, 0]])
    orisecret = torch.from_numpy(orisecret).float()
    orisecret = orisecret.to(device)

    print(orisecret.shape)
    print(orisecret)

    with torch.no_grad():
        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for filename in files_list:
                image = cv2.imread(filename, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
                image = cv2.resize(image, (width, height))  # 400x400 以匹配 encoder
                image = transforms.ToTensor()(image).unsqueeze(0).to(device)

                inputs = (orisecret, image)
                residual = encoder(inputs)

                encoded_image = residual + image
                encoded_image = torch.clamp(encoded_image, 0, 1)
                result_encoded_img = transforms.ToPILImage()(encoded_image.squeeze(0)).resize((512, 512))  # 变为 512x512

                residual = residual + .5
                result_residual = transforms.ToPILImage()(residual.squeeze(0)).resize((512, 512))  # 变为 512x512

                save_name = filename.split('/')[-1].split('.')[0]
                save_path_encoded_img = os.path.join(args.save_dir, f'{save_name}_encoded.png')
                save_path_residual = os.path.join(args.save_dir, f'{save_name}_residual.png')
                result_encoded_img.save(save_path_encoded_img)
                result_residual.save(save_path_residual)


if __name__ == "__main__":
    main()
