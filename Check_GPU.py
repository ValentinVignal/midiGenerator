import tensorflow as tf
from tensorflow.python.client import device_lib
import sys
import argparse

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    # return [x.name for x in local_device_protos if x.device_type == 'GPU']
    return local_device_protos

def main():

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('-l', '--list-local-devices', default=False, action='store_true',
                        help='Use tf.python.client.device_lib.list_local_devices()')
    parser.add_argument('-g', '--gpu-device-name', default=False, action='store_true',
                        help='Use tf.test.gpu_device_name()')

    args = parser.parse_args()

    if args.list_local_devices:
        name = get_available_gpus()
        print("Devices:{0}".format(name))
    if args.gpu_device_name:
        print('GPU : {0}'.format(tf.test.gpu_device_name()))


    # Try using gpu


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
