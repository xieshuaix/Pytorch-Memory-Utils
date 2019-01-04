import gc
import datetime
import pynvml
import os

import torch
import numpy as np
from operator import itemgetter


class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        frame: a frame to detect current py-file runtime
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """

    def __init__(self, frame, detail=True, path='', verbose=False, device=0):
        self.frame = frame
        self.print_detail = detail
        self.last_summary = set()
        self.file_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '-gpu_mem_track.txt'
        self.gpu_profile_fn = os.path.join(path, self.file_name)
        self.verbose = verbose
        self.begin = True
        self.device = device
        self.memory_min = 0.001  # Tensors of allocated memory smaller than this value are not logged

        self.func_name = frame.f_code.co_name
        self.filename = frame.f_globals["__file__"]
        if (self.filename.endswith(".pyc") or
                self.filename.endswith(".pyo")):
            self.filename = self.filename[:-1]
        self.module_name = self.frame.f_globals["__name__"]
        self.curr_line = self.frame.f_lineno

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    @staticmethod
    def get_tensor_info(tensor_list):
        type_list = []
        shape_list = []
        size_list = []
        n = len(tensor_list)
        for t in tensor_list:
            type_list.append("{}, {}".format(str(type(t)), str(t.dtype)))
            shape_list.append(str(tuple(t.size())))
            size_list.append(t.nelement() * t.element_size() / 1e6)

        return [type_list, shape_list, size_list]

    @staticmethod
    def distinguish_param_cache(tensor_list):
        param_indices = [i for i, v in enumerate(tensor_list[0]) if str(v).find('param') != -1]
        cache_indices = [i for i in range(n) if i not in param_indices]
        param_list = [[tensor_list[0][i] for i in param_indices],
                      [tensor_list[1][i] for i in param_indices],
                      [tensor_list[2][i] for i in param_indices],
                      [tensor_list[3][i] for i in param_indices]
                      ]
        cache_list = [[tensor_list[0][i] for i in cache_indices],
                      [tensor_list[1][i] for i in cache_indices],
                      [tensor_list[2][i] for i in cache_indices],
                      [tensor_list[3][i] for i in cache_indices]
                      ]

        return param_list, cache_list

    def track(self, header=None):
        """
        Track the GPU memory usage
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.curr_line = self.frame.f_lineno
        where_str = self.module_name + ' ' + self.func_name + ':' + ' line ' + str(self.curr_line)

        with open(self.gpu_profile_fn, 'a+') as f:
            if self.begin:
                f.write('GPU Memory Tracker\n')
                self.begin = False
            if header is not None:
                f.write("\n{}\n".format(header))

            time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            f.write("{} | @ {:<50} | Total Used Memory:{:>7.1f}MB\n".format(time, where_str, meminfo.used / 1000 ** 2))

            if self.print_detail is True:
                info_list = self.get_tensor_info([t for t in self.get_tensors()])
                info_tuple_list = [info for info in zip(*info_list)]
                new_summary = {info + (info_tuple_list.count(info), ) for info in info_tuple_list}
                if len(new_summary - self.last_summary) != 0:
                    increment = sorted(list(new_summary - self.last_summary), key=itemgetter(2), reverse=True)
                    for t, s, b, c in increment:
                        if b > self.memory_min:
                            f.write(
                                '+ | {:>2} * Size:{:<20} | Memory: {:.3f}MB | {:<20}\n'.format(c, s, b * c, t))
                    decrement = sorted(list(new_summary - self.last_summary), key=itemgetter(2), reverse=False)
                    for t, s, b, c in decrement:
                        if b > self.memory_min:
                            f.write(
                                '- | {:>2} * Size:{:<20} | Memory: {:.3f}MB | {:<20}\n'.format(c, s, b * c, t))
                else:
                    f.write('No change in tensor shapes, sizes and types\n')
                self.last_summary = new_summary

        pynvml.nvmlShutdown()
