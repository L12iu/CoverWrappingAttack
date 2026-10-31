import glob
import bchlib
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import sys
from torchvision import transforms
from model import StegaStampDecoder

# BCH纠错码配置参数，需与编码端保持一致
BCH_POLYNOMIAL = 137
BCH_BITS = 5


def get_secret_acc(secret_true, secret_pred):
    # 比特精确度,和秘密消息正确率
    # 将预测值和真值从GPU转移到CPU，避免混合设备运算报错
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)  # 四舍五入，将概率值转为0/1二值
    # 计算每个样本中正确预测的比特数
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    # 计算字符串准确率：完全正确的样本数 / 总样本数
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    # 计算比特准确率：所有正确比特数 / 总比特数
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()  # numel()函数：返回数组中元素的个数
    return bit_acc, str_acc


def main(ti):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth', )
    # parser.add_argument('--model', type=str, default='/opt/data/mingjin/pycharm/Net/exp/rihoop_dir/saved_models/demo1/')

    # parser.add_argument('--image', type=str,
                        # default='/opt/data/helin/Code/rihoop_dir/result/rihoop/test/000000274708_hidden.png')
    parser.add_argument('--image', type=str,
                        default='/opt/data/jiang/cross/CRoSS/image_jiang/hide_encoded.png')

    # parser.add_argument('--images_dir', type=str, default='/opt/data/helin/Code/rihoop_dir/result/rihoop/test/')
    parser.add_argument('--secret_size', type=int, default=100)
    args = parser.parse_args()
    # args.images_dir=args.images_dir+'result_{}'.format(ti)
    # 根据命令行参数确定待处理的文件列表
    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    # 指定GPU设备
    device = torch.device('cuda:0')
    # 初始化解码器模型
    decoder = StegaStampDecoder(args)
    # 加载模型权重，先映射到CPU避免显存不足
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    # 加载解码器权重
    decoder.load_state_dict(checkpoint['model_decoder'])
    # 将模型移动到GPU
    decoder.to(device)
    # 设置为评估模式，关闭Dropout等训练专用层
    decoder.eval()
    # 硬编码的原始秘密信息真值（100位二进制），用于计算准确率
    orisecret = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                        0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                        0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                        0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                        1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                        0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                        0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                        0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                        1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                        ]])
    orisecret = torch.from_numpy(orisecret).float()
    orisecret = orisecret.to(device)
    # decoder = torch.load(args.model).cuda()

    # bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    sum_bit=0
    number=0
    print('==========================================')
    for filename in files_list:
        number+=1
        # print(filename)
        # 读取图片并转换颜色空间 BGR -> RGB
        image = cv2.imread(filename, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
        # 调整图片尺寸至400x400，匹配模型输入要求
        image = cv2.resize(image, (400,400))
        # 转换为Tensor并增加批次维度
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device)
        # image = Image.open(filename).convert("RGB")
        # image = np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32)
        # image /= 255.
        # 提取文件名作为保存标识
        save_name = filename.split('/')[-1].split('.')[0]
        # 模型推理并四舍五入得到二值秘密信息
        secret = torch.round(decoder(image))
        # print(secret)
        # print(secret.shape)
        # 计算比特准确率
        bit,_=get_secret_acc(secret,orisecret)
        print('{} bit:{}'.format(save_name,bit))
        # sum_bit+=bit
    # return sum_bit / number


if __name__ == "__main__":
     main('medium_new')
    # bitavg2 = main('strong_new')
    # bitavg3 = main('weak')
    # bitavg4 = main('left60')
    # bitavg5 = main('right10')
    # bitavg6 = main('right30')
    # bitavg7 = main('right60')
    # print("bit_avg{}:{}".format('medium_new', bitavg1))
    # print("bit_avg{}:{}".format('strong_new', bitavg2))
    # print("bit_avg{}:{}".format('weak', bitavg3))
    # # print("bit_avg{}:{}".format(4, bitavg4))
    # print("bit_avg{}:{}".format(5, bitavg5))
    # print("bit_avg{}:{}".format(6, bitavg6))
    # print("bit_avg{}:{}".format(7, bitavg7))