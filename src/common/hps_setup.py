    # -*- coding: utf-8 -*-
"""
General hyperparameters for hyperlooping and slurm setup
@author: thomas
"""

import copy       

def get_hps_setup():
    ''' Hyperparameter settings '''
    return HParams(      
        # General
        game = 'None', # Environment name
        name = 'None', # Name of experiment

        # Slurm parameters
        slurm = False,
        slurm_qos = 'short',        
        slurm_time = '3:59:59',
        cpu_per_task = 2, 
        mem_per_cpu = 2048,
        distributed = False,
        n_jobs = 16, # distribute over n_jobs processes
        
        # Hyperparameter looping
        n_rep = 1,
        rep = 0, # repetition index
        loop_hyper = False,
        item1 = None,
        seq1 = [None],
        item2 = None,
        seq2 = [None],
        item3 = None,
        seq3 = [None],
        item4 = None,
        seq4 = [None],
        )

def hps_to_list(hps):
    out=[]
    hps_dict = copy.deepcopy(hps.__dict__)
    try:
        del hps_dict['_items']
    except:
        pass
    for k,v in hps_dict.items():
        if type(v) == list:
            v='+'.join(str(x) for x in v)
        if not (v is None or v == 'None'): # should not write the default hyperloop settings
            out.append('{}={}'.format(k,v))
    out.sort()
    return ','.join(out)

def hps_to_dict(hps):
    hps_dict = copy.deepcopy(hps.__dict__)
    try:
        del hps_dict['_items']
    except:
        pass
    return hps_dict
        
class HParams(object):

    def __init__(self, **kwargs):
        self._items = {}
        for k, v in kwargs.items():
            self._set(k, v)

    def _set(self, k, v):
        self._items[k] = v
        setattr(self, k, v)
        
    def _get(self,k):
        return self._items[k]
        
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__

    def parse(self, str_value,hps_extra=None):
        hps = HParams(**self._items)
        for entry in str_value.strip().split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError("Unable to parse: %s" % entry)
            try:
                default_value = hps._items[key]
            except:
                print('Cant parse key {}, skipping'.format(key))
                continue
            if isinstance(default_value, bool):
                hps._set(key, value.lower() == "true")
            elif isinstance(default_value, int):
                hps._set(key, int(value))
            elif default_value is None and value == 'None':
                hps._set(key, None)
            elif isinstance(default_value, float):
                hps._set(key, float(value))
            elif isinstance(default_value, list):
                value = value.split('+')
                default_inlist = hps._items[key][0]
                if key == 'seq1':
                    if hps_extra is not None:
                        default_inlist = hps_extra._items[hps._items['item1']]
                    else:
                        default_inlist = hps._items[hps._items['item1']]                        
                if key == 'seq2':
                    if hps_extra is not None:
                        default_inlist = hps_extra._items[hps._items['item2']]
                    else:
                        default_inlist = hps._items[hps._items['item2']]                        
                if key == 'seq3':
                    if hps_extra is not None:
                        default_inlist = hps_extra._items[hps._items['item3']]
                    else:
                        default_inlist = hps._items[hps._items['item3']]                        
                if isinstance(default_inlist, bool):
                    hps._set(key, [i.lower() == "true" for i in value])
                elif isinstance(default_inlist, int):
                    hps._set(key, [int(i) for i in value])
                elif isinstance(default_inlist, float):
                    hps._set(key, [float(i) for i in value])
                else:
                    hps._set(key,value) # string
            else:
                hps._set(key, value)
        return hps
