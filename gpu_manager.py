# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:41:55 2017

@author: Quantum Liu
"""
'''
Example:
gm=GPUManager()
with torch.cuda.device(gm.auto_choice())

Or:
gm=GPUManager()
torch.cuda.set_device(gm.auto_choice())
'''

import os
import torch


def check_gpus():
    """
    GPU available check
    http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    """
    if not torch.cuda.is_available():
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif 'NVIDIA System Management' not in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True


if check_gpus():
    def parse(line, qargs):
        """
        line:
            a line of text
        qargs:
            query arguments
        return:
            a dict of gpu infos
        Parsing a line of csv format text returned by nvidia-smi
        """
        args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']

        def power_manage_enable(v):
            return 'Not Support' not in v

        def to_numeric(v):
            v = v.strip().upper().replace('MIB', '').replace('W', '').replace(' ', '')
            if v.isdigit():
                return int(v)
            elif v.replace('.', '').isdigit():
                return float(v)
            elif 'ERROR' in v or 'UNKNOWN' in v:
                return 'NA'
            else:
                return v

        def process(k, v):
            return (to_numeric(v) if power_manage_enable(v) else 1) if k in args else v.strip()

        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


    def query_gpu(qargs=[]):
        """
        qargs:
            query arguments
        return:
            a list of dict
        Querying GPUs info
        """
        qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit'] + qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [parse(line, qargs) for line in results]


    def by_power(d):
        """
        helper function fo sorting gpus by power
        """
        power_info = (d['power.draw'], d['power.limit'])
        if any(v == 1 for v in power_info):
            print('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw']) / d['power.limit']


    class GPUManager:
        """
        qargs:
            query arguments
        A manager which can list all available GPU devices
        and sort them and choice the most free one.Unspecified
        ones pref.
        GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
        最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
        优先选择未指定的GPU。
        """

        def __init__(self, qargs=[]):
            self.qargs = qargs
            self.gpus = query_gpu(qargs)
            for gpu in self.gpus:
                gpu['specified'] = False
            self.gpu_num = len(self.gpus)

        @staticmethod
        def _sort_by_memory(gpus, by_size=False):
            if by_size:
                return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
            else:
                return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

        @staticmethod
        def _sort_by_power(gpus):
            return sorted(gpus, key=by_power)

        @staticmethod
        def _sort_by_custom(gpus, key, reverse=False, qargs=[]):
            if isinstance(key, str) and (key in qargs):
                return sorted(gpus, key=lambda d: d[key], reverse=reverse)
            if isinstance(key, type(lambda a: a)):
                return sorted(gpus, key=key, reverse=reverse)
            raise ValueError(
                "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

        def find(self, mode=0):
            """
            mode:
                0:(default)sorted by free memory size
            return:
                a TF device object
            Auto choice the freest GPU device,not specified
            ones
            """
            for old_info, new_info in zip(self.gpus, query_gpu(self.qargs)):
                old_info.update(new_info)
            unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

            if mode == 0:
                print('Finding best w.r.t available memory...')
                chosen_gpu = self._sort_by_memory(unspecified_gpus, True)[0]
            elif mode == 1:
                print('Finding best w.r.t portion of available memory...')
                chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
            elif mode == 2:
                print('Finding best GPU w.r.t. power...')
                chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
            else:
                print('Unsupported mode, finding GPU w.r.t portion of available memory...')
                chosen_gpu = self._sort_by_memory(unspecified_gpus)[0]
            chosen_gpu['specified'] = True
            index = chosen_gpu['index']
            print('Using GPU {i} (mem in MB, power in W):\n\t{info}'.format(i=index, info='\n\t'.join(
                [str(k) + ': ' + str(v) for k, v in sorted(chosen_gpu.items())])))
            return int(index)
else:
    raise ImportError('GPU available check failed')
