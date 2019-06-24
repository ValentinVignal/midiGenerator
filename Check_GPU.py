import tensorflow as tf
from tensorflow.python.client import device_lib
import sys


def main():
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        # return [x.name for x in local_device_protos if x.device_type == 'GPU']
        return local_device_protos
    name = get_available_gpus()
    print("GPUS:{}".format(name))

    # Try using gpu


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
