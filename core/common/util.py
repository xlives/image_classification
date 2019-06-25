"""
Various utilities that don't fit anwhere else.
"""
from itertools import zip_longest, islice
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Iterable, Iterator
import importlib
import json
import logging
import pkgutil
import random
import subprocess
import sys
import os

try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None

import torch
import numpy

# This base import is so we can refer to allennlp.data.Token in `sanitize()` without creating
# circular dependencies.
from core.common.checks import log_pytorch_version_info
from core.common.tqdm import Tqdm
from core.common.tee_logger import TeeLogger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

JsonDict = Dict[str, Any]  # pylint: disable=invalid-name


def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):  # pylint: disable=no-member
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                         "If this is your own custom class, add a `to_json(self)` method "
                         "that returns a JSON-like object.")

def group_by_count(iterable: List[Any], count: int, default_value: Any) -> List[List[Any]]:
    """
    Takes a list and groups it into sublists of size ``count``, using ``default_value`` to pad the
    list at the end if the list is not divisable by ``count``.

    For example:
    >>> group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 0)
    [[1, 2, 3], [4, 5, 6], [7, 0, 0]]

    This is a short method, but it's complicated and hard to remember as a one-liner, so we just
    make a function out of it.
    """
    return [list(l) for l in zip_longest(*[iter(iterable)] * count, fillvalue=default_value)]

A = TypeVar('A')

def lazy_groups_of(iterator: Iterator[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterator and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    return iter(lambda: list(islice(iterator, 0, group_size)), [])


def add_noise_to_dict_values(dictionary: Dict[A, float], noise_param: float) -> Dict[A, float]:
    """
    Returns a new dictionary with noise added to every key in ``dictionary``.  The noise is
    uniformly distributed within ``noise_param`` percent of the value for every value in the
    dictionary.
    """
    new_dict = {}
    for key, value in dictionary.items():
        noise_value = value * noise_param
        noise = random.uniform(-noise_value, noise_value)
        new_dict[key] = value + noise
    return new_dict


def prepare_global_logging(serialization_dir: str, file_friendly_logging: bool) -> logging.FileHandler:
    """
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Note that this function does not set the logging level, which is set in ``allennlp/run.py``.

    Parameters
    ----------
    serialization_dir : ``str``, required.
        The directory to stream logs to.
    file_friendly_logging : ``bool``, required.
        Whether logs should clean the output to prevent carriage returns
        (used to update progress bars on a single terminal line). This
        option is typically only used if you are running in an environment
        without a terminal.

    Returns
    -------
    ``logging.FileHandler``
        A logging file handler that can later be closed and removed from the global logger.
    """

    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    Tqdm.set_slower_interval(file_friendly_logging)
    std_out_file = os.path.join(serialization_dir, "stdout.log")
    sys.stdout = TeeLogger(std_out_file, # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), # type: ignore
                           sys.stderr,
                           file_friendly_logging)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(stdout_handler)

    return stdout_handler

def cleanup_global_logging(stdout_handler: logging.FileHandler) -> None:
    """
    This function closes any open file handles and logs set up by `prepare_global_logging`.

    Parameters
    ----------
    stdout_handler : ``logging.FileHandler``, required.
        The file handler returned from `prepare_global_logging`, attached to the global logger.
    """
    stdout_handler.close()
    logging.getLogger().removeHandler(stdout_handler)

    if isinstance(sys.stdout, TeeLogger):
        sys.stdout = sys.stdout.cleanup()
    if isinstance(sys.stderr, TeeLogger):
        sys.stderr = sys.stderr.cleanup()


def import_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    sys.path.append('.')

    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, '__path__', [])
    path_string = '' if not path else path[0]

    # walk_packages only finds immediate children, so need to recurse.
    for module_finder, name, _ in pkgutil.walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path,
        # `pkgutil.walk_packages` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)


def peak_memory_mb() -> float:
    """
    Get peak memory usage for this process, as measured by
    max-resident-set size:

    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size

    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return 0.0

    # TODO(joelgrus): For whatever, our pinned version 0.521 of mypy does not like
    # next line, but later versions (e.g. 0.530) are fine with it. Once we get that
    # figured out, remove the type: ignore.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == 'darwin':
        # On OSX the result is in bytes.
        return peak / 1_000_000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1_000

def gpu_memory_mb() -> Dict[int, int]:
    """
    Get the current GPU memory usage.
    Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    ``Dict[int, int]``
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
        Returns an empty ``dict`` if GPUs are not available.
    """
    # pylint: disable=bare-except
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                          '--format=csv,nounits,noheader'],
                                         encoding='utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return {gpu: memory for gpu, memory in enumerate(gpu_memory)}
    except FileNotFoundError:
        # `nvidia-smi` doesn't exist, assume that means no GPU.
        return {}
    except:
        # Catch *all* exceptions, because this memory check is a nice-to-have
        # and we'd never want a training run to fail because of it.
        logger.exception("unable to check gpu_memory_mb(), continuing")
        return {}


def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)

def is_lazy(iterable: Iterable[A]) -> bool:
    """
    Checks if the given iterable is lazy,
    which here just means it's not a list.
    """
    return not isinstance(iterable, list)

def get_frozen_and_tunable_parameter_names(model: torch.nn.Module) -> List:
    frozen_parameter_names = []
    tunable_parameter_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen_parameter_names.append(name)
        else:
            tunable_parameter_names.append(name)
    return [frozen_parameter_names, tunable_parameter_names]

def dump_metrics(file_path: str, metrics: Dict[str, Any], log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)

def flatten_filename(file_path: str) -> str:
    return file_path.replace('/', '_SLASH_')
