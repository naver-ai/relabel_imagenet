"""Python wrapper for experiment configurations.
"""
__author__ = "sanghyuk-c"

import collections.abc
import pickle

import munch
import yaml
from yaml.error import YAMLError

try:
    import torch
except ImportError:
    pass

from utils.utils import DEFAULT_CONFIG_STR
from utils.utils import str2bool

_SEPARATOR = '.'


def _print(msg, verbose):
    """A simple print wrapper.
    """
    if verbose:
        print(msg)


def _loader(config_path: str,
            verbose: bool = False) -> dict:
    """A simple serializer loader.
    Examples
    --------
    >>> _loader('test.json')
    """
    with open(config_path, 'r') as fin:
        try:
            return yaml.load(fin)
        except YAMLError:
            _print('failed to load from yaml. Try pickle loader', verbose)

    with open(config_path, 'rb') as fin:
        try:
            return pickle.load(fin)
        except TypeError:
            _print('failed to load from pickle. Try torch loader', verbose)
        try:
            return torch.load(fin)
        except TypeError:
            _print('failed to load from pickle. '
                   'Please check your configuration again.', verbose)

    raise TypeError(
        'config_path should be serialized by [yaml, json, pickle, torch pth]')


def _recursively_get_value_dict(dict_object, key):
    list_of_key_hierarchy = key.split(_SEPARATOR)
    if len(list_of_key_hierarchy) == 1:
        return dict_object[key]
    # if len(list_of_key_hierarchy) >= 2
    child_dict_object = dict_object[list_of_key_hierarchy[0]]
    child_key = _SEPARATOR.join(list_of_key_hierarchy[1:])
    return _recursively_get_value_dict(child_dict_object, child_key)


def _recursively_set_value_dict(dict_object, key, val):
    list_of_key_hierarchy = key.split(_SEPARATOR)
    # print(dict_object, key, val)
    if len(list_of_key_hierarchy) == 1:
        dict_object[key] = val
        return
    # if len(list_of_key_hierarchy) >= 2
    child_dict_object = dict_object[list_of_key_hierarchy[0]]
    child_key = _SEPARATOR.join(list_of_key_hierarchy[1:])
    _recursively_set_value_dict(child_dict_object, child_key, val)


def _update_dict_with_another(d, u):
    """
    https://stackoverflow.com/questions/3232943/
    update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update_dict_with_another(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_config(config_fname: str,
                 strict_cast: bool = True,
                 verbose: bool = True,
                 **kwargs) -> munch.Munch:
    """Parse the given configuration file with additional options to overwrite.
    Parameters
    ----------
    config_fname: str
        A configuration file defines the structure of the configuration.
        The file should be serialized by any of [yaml, json, pickle, torch].
    strict_cast: bool, optional, default=True
        If True, the overwritten config values will be casted as the
        original type.
    verbose: bool, optional, default=False
    kwargs: optional
        If specified, overwrite the current configuration by the given keywords.
        For the multi-depth configuration, "__" is used
        for the default delimiter.
        The keys in kwargs should be already defined by config_fname
         (otherwise it will raise KeyError).
        Note that if `strict_cast` is True, the values in kwargs will be casted
         as the original type defined in the configuration file.
    Returns
    -------
    config: munch.Munch
        A configuration file, which provides attribute-style access.
        See `Munch <https://github.com/Infinidat/munch>` project
        for the details.
    Examples
    --------
    >>> # simple_config.json => {"opt1": {"opt2": 1}, "opt3": 0}
    >>> config = parse_config('simple_config.json')
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2),
              type(config.opt3))
    2 1 <class 'int'> <class 'int'>
    >>> config = parse_config('simple_config.json', opt1__opt2=2, opt3=1)
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2),
              type(config.opt3))
    2 1 <class 'int'> <class 'int'>
    >>> parse_config('test.json', **{'opt1__opt2': '2', 'opt3': 1.0})
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2),
              type(config.opt3))
    2 1 <class 'int'> <class 'int'>
    >>> parse_config('test.json', **{'opt1__opt2': '2', 'opt3': 1.0},
                     strict_cast=False)
    >>> print(config.opt1.opt2, config.opt3, type(config.opt1.opt2),
              type(config.opt3))
    2 1.0 <class 'str'> <class 'float'>
    """
    base_config = _loader('configs/base.yaml', verbose)
    additional_config = _loader(config_fname, verbose)

    config = _update_dict_with_another(base_config, additional_config)

    for arg_key, arg_val in kwargs.items():
        if arg_val == DEFAULT_CONFIG_STR:
            continue
        if strict_cast:
            typecast = type(_recursively_get_value_dict(config, arg_key))
            if typecast is bool and isinstance(arg_val, str):
                arg_val = str2bool(arg_val)
            _recursively_set_value_dict(config, arg_key, typecast(arg_val))
        else:
            _recursively_set_value_dict(config, arg_key, arg_val)

    config = munch.munchify(config)
    _print(config, verbose)
    return config
