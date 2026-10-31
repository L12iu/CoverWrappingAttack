#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 小何一定行
import argparse

import bchlib
import cv2
import numpy as np
import torch

# required参数可以设置该参数是否必需。
parser = argparse.ArgumentParser()
parser.add_argument('--detector_model', type=str,
                    default='/opt/data/mingjin/pycharm/Exercise/stegastamp/StegaStamp-pytorch/result_model/result_model_BiSeNet.pth')
parser.add_argument('--decoder_model', type=str,
                    default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth')
parser.add_argument('--video', type=str,
                    default='/opt/data/helin/Code/StegaStamp-pytorch/Test/test_video.mp4')
parser.add_argument('--secret_size', type=int, default=100)
parser.add_argument('--save_video', type=str, default='/opt/data/helin/Code/StegaStamp-pytorch/Test')
# action='store_true' 表示有设置--visualize_detector参数时为true,没设置时为false
# 且如果指定参数出现，则action_true/false起作用，如果参数不出现则default起作用。
parser.add_argument('--visualize_detector', action='store_true', help='Visualize detector mask output')
args = parser.parse_args()

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def get_intersect(p1, p2, p3, p4):
    s = np.vstack([p1, p2, p3, p4])
    h = np.hstack((s, np.ones((4, 1))))
    """
    np.coss()求叉乘：
    a=(X1,Y1,Z1),b=(X2,Y2,Z2),
    a×b=（Y1Z2-Y2Z1,Z1X2-Z2X1,X1Y2-X2Y1）
    (1,2,3)×(4,5,6)=(12-15,12-6,5-8)=（-3,6,-3）
    """
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        print('invalid')
        return (0, 0)
    return (x / z, y / z)


def poly_area(poly):  # 多边形面积？
    return 0.5 * np.abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    """
    numpy.diff(a, n=1,axis=-1)沿着指定轴计算第N维的离散差值
    参数：
    a：输入矩阵
    n：可选，代表要执行几次差值
    axis：默认是最后一个
    """
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def main():
    # Initializing network
    detector = torch.load(args.detector_model)
    decoder = torch.load(args.decoder_model)

    # cap = cv2.VideoCapture(0)参数是0,表示打开笔记本的内置摄像头,参数是视频文件路径则打开视频,如cap=cv2.VideoCapture("…/test.avi")
    cap = cv2.VideoCapture(args.video)
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # ret, frame = cap.read()
    # cap.read()按帧读取视频，ret、frame是获cap.read()方法的两个返回值。
    # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
    # frame就是每一帧的图像，是个三维矩阵(h*w*c)。这里就是视频一开始的那帧图
    ret, frame = cap.read()
    f_height, f_width = frame.shape[0:2]

    if args.save_video is not None:
        # 保存视频
        # 第一个参数是要保存的文件的路径
        # fourcc 指定编码器
        # fps 要保存的视频的帧率
        # frameSize 要保存的文件的画面尺寸
        fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.save_video, fourcc1, 30.0, (f_width, f_height))

    while (True):
        ret, frame = cap.read()
        if frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detector_image_input = cv2.resize(frame_rgb, (1024, 1024))
        # (1, 1024, 1024, 3)
        detector_image_input = np.expand_dims(np.float32(detector_image_input), axis=0) / 255.0

        output_image = detector(detector_image_input)  # 语义分割图的大小是b*h*w*1
        output_image = np.array(output_image[0, :, :, :])
        # 找出颜色通道里颜色值最大的在哪个通道-->0
        output_image = x = np.argmax(output_image, axis=-1)  # h*w

        color_codes = np.array([[255, 255, 255], [0, 0, 0]])
        out_vis_image = color_codes[output_image.astype(int)]  # 黑白轮廓图？

        mask_im = cv2.resize(np.float32(out_vis_image), (f_width, f_height))  # 变回1024*1024
        if args.visualize_detector:
            mask_vis = mask_im.astype(np.uint8)
        """
        # 利用cv2.findContours()函数来查找检测物体的轮廓,且findcontours函数会“原地”修改输入的图像
        # 接受的参数为二值图,即黑白的（不是灰度图）,所以读取的图像要先转成灰度的,再转成二值图
        cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])
        cv2.findContours()函数返回两个值contours,hierarchy,一个是轮廓本身(一个list),还有一个是每条轮廓对应的属性
        """
        contours, _ = cv2.findContours(cv2.cvtColor(mask_im, cv2.COLOR_BGR2GRAY).astype(np.uint8), 1, 2)
        extrema = np.zeros((8, 2))
        corners = np.zeros((4, 2))

        for cnt in contours:
            # 计算轮廓区域面积
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            """
            hull = cv2.convexHull(points, clockwise, returnpoints) :计算凸包
            其中，各个参数的意义如下：
            hull : 输出凸包结果，n * 1 *2 数据结构，n为外包围圈点数
            points: 输入的坐标点，通常为1* n * 2 结构，n为所有的坐标点的数目
            clockwise：转动方向，TRUE为顺时针，否则为逆时针；
            returnPoints：默认为TRUE，返回凸包上点的坐标，如果设置为FALSE，会返回与凸包点对应的轮廓上的点。
            """
            hull = cv2.convexHull(cnt)
            if len(hull) < 4:
                continue

            if args.visualize_detector:
                """
                绘制多边形或折线段
                polylines(image, pts, isClosed, color[, thickness[, lineType[, shift]]])
                参数：image：要在上面画多边形的图像
                     pts：包含多边形上点的数组,pts参数外面要用[]括起来,pts参数中坐标的顺序为 (x, y)
                     isClosed：标志,决定所绘制的多边形是否闭合.若为 True,则画若干个闭合多边形;若为 False,则画一条连接所有点的折线
                     color：多边形颜色
                     thickness：多边形线的粗细
                     lineType：多边形线的类型
                     shift：坐标精确到小数点后第几位
                """
                cv2.polylines(mask_vis, np.int32([corners]), thickness=6, color=(100, 100, 250), isClosed=True)

                extrema[0, :] = hull[np.argmax(hull[:, 0, 0]), 0, :]  # 取x坐标最大的点
                extrema[1, :] = hull[np.argmax(hull[:, 0, 0] + hull[:, 0, 1]), 0, :]
                extrema[2, :] = hull[np.argmax(hull[:, 0, 1]), 0, :]  # 取y坐标最大的点
                extrema[3, :] = hull[np.argmax(-hull[:, 0, 0] + hull[:, 0, 1]), 0, :]
                extrema[4, :] = hull[np.argmax(-hull[:, 0, 0]), 0, :]  # 取x坐标最小的点
                extrema[5, :] = hull[np.argmax(-hull[:, 0, 0] - hull[:, 0, 1]), 0, :]
                extrema[6, :] = hull[np.argmax(-hull[:, 0, 1]), 0, :]
                extrema[7, :] = hull[np.argmax(hull[:, 0, 0] - hull[:, 0, 1]), 0, :]

                extrema_lines = extrema - np.roll(extrema, shift=1, axis=0)
                extrema_len = extrema_lines[:, 0] ** 2 + extrema_lines[:, 1] ** 2  # 点点间的距离
                line_idx = np.sort(extrema_len.argsort()[-4:])
                for c in range(4):
                    p1 = extrema[line_idx[(c - 1) % 4], :]
                    p2 = extrema[(line_idx[(c - 1) % 4] - 1) % 8, :]
                    p3 = extrema[line_idx[c], :]
                    p4 = extrema[(line_idx[c] - 1) % 8, :]
                    corners[c, :] = get_intersect(p1, p2, p3, p4)

                new_area = poly_area(corners)  # 计算多边形面积
                if new_area / area > 1.5:
                    continue

                corners = order_points(corners)
                corners_full_res = corners

                # 计算单应性矩阵,进行透视扭曲
                pts_dst = np.array([[0, 0], [399, 0], [399, 399], [0, 399]])
                h, status = cv2.findHomography(corners_full_res, pts_dst)
                try:
                    warped_im = cv2.warpPerspective(frame_rgb, h, (400, 400))
                    w_im = warped_im.astype(np.float32)
                    w_im /= 255.
                except:
                    continue

                for im_rotation in range(4):

                    # np.rot90(A,k) :矩阵A逆时针旋转90°或90°的k倍
                    w_rotated = np.rot90(w_im, im_rotation)

                    recovered_secret = decoder(w_rotated)
                    recovered_secret = list(recovered_secret)
                    recovered_secret = [int(i) for i in recovered_secret]

                    packet_binary = "".join([str(bit) for bit in recovered_secret[:96]])
                    footer = recovered_secret[96:]
                    if np.sum(footer) > 0:
                        continue

                    packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
                    packet = bytearray(packet)

                    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

                    bitflips = bch.decode_inplace(data, ecc)

                    if bitflips != -1:
                        print('Num bits corrected: ', bitflips)
                        try:
                            code = data.decode("utf-8")
                        except:
                            continue
                        color = (100, 250, 100)
                        # 绘制多边形或折线段
                        cv2.polylines(frame, np.int32([corners]), thickness=6, color=color, isClosed=True)
                        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体：FONT_HERSHEY_SIMPLEX=0
                        """
                        cv2.putText(image, str(i), (123,456)), font, 2, (0,255,0), 3)
                        各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                        """
                        im = cv2.putText(frame, code, tuple((corners[0, :] + np.array([0, -15])).astype(np.int)), font,
                                         1, (0, 0, 0), 2, cv2.LINE_AA)

        if args.save_video is not None:
            out.write(frame)
        else:
            cv2.imshow('frame', frame)
            if args.visualize_detector:
                cv2.imshow('detector_mask', mask_vis)
            cv2.waitKey(1)

    cap.release()
    if args.save_video:
        out.release()


if __name__ == "__main__":
    main()
