import numpy as np

bit_array = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,
                      1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
                      0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                      0, 0, 1, 0])

# 将 bit_array 转换为字节
byte_array = np.packbits(bit_array)

# 尝试解码为 utf-8
try:
    decoded_string = byte_array.tobytes().decode('utf-8')
    print("Decoded string:", decoded_string)
except UnicodeDecodeError:
    print("Decode Error")
