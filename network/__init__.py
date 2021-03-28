import logging
import socket
import pickle
import struct
import utils
from PIL import Image
import cv2
import numpy as np

def recvall(sock, count):
    """
    由于recv函数可能不会返回等于参数大小的内容，需要不断接收确保接收到了足够长的内容
    :param sock: socket对象
    :param count: 需要接受的字节数
    :return: 返回等于count大小的缓冲区
    """
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def send_configuration(s,obj):
    """
    :param s: socket对象
    :param obj: 准备发送的对象，结构为字典{‘GT’:高清列表，‘LQs':低清列表}训练的时候根据收到的sample块重建训练
    :return: 无
    """
    data=pickle.dumps(obj)
    #print(len(data))
    #print(data)
    data_len=struct.pack('l',len(data))
    s.send(data_len)
    s.send(data)

def handel_stream(csock):
    """
    :param csock: socket对象
    :return: 返回两个值，isSample，data，前者为false，则为解码的图像，否则为样本
    """
    logger = logging.getLogger('base')
    #c_reader = csock.makefile("rb")  # 此处必须使用makefile函数才行
    #data_len = c_reader.read(struct.calcsize('l'))
    data_len=recvall(csock,struct.calcsize('l'))
    assert len(data_len) == struct.calcsize('l'), 'data_len is less than {}, transport error!'.format(
        struct.calcsize('l'))
    data_len = struct.unpack('l', data_len)[0]
    #logger.info(data_len)
    #data = c_reader.read(data_len)
    if data_len>0:
        data=recvall(csock,data_len)
        data = pickle.loads(data)
        is_sample = data['sample']
    else:
        data=None
        is_sample=False
    return is_sample, data


# def get_jpeg_stream(csock, data_len):
#     logger = logging.getLogger('base')
#     string_data = recvall(csock, data_len)
#     logger.info('string_data len is:{}'.format(data_len))
#     #string_data=csock.recv(data_len)
#     decimg=utils.decode_jpeg(string_data)
#     if decimg is None:
#         logger.info('dec jpeg none!')
#     return decimg
