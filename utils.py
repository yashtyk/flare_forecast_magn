import sklearn
import yaml
from pathlib import Path

def tss(tp: int, fp: int, tn: int, fn: int) -> float:
    # True Skill Statistic
    # also computed as sensitivity + specificity - 1
    # in [-1,1], best at 1, no skill at 0
    # Always majority class: 0
    # Random: 0
    return tp / (tp + fn) + tn / (tn + fp) - 1

def hss(tp: int, fp: int, tn: int, fn: int) -> float:
    # Heidke Skill Score - computation inspired by hydrogo/rainymotion
    # in [-inf,1], best at 1
    # Always majority class: 0
    # Random: 0
    return (2 * (tp * tn - fn * fp)) / (fn ** 2 + fp ** 2 + 2 * tp * tn + (fn + fp) * (tp + tn))

def write_yaml(file: Path, data: dict):
    """
    Write dict to yaml file (yaml version 1.2)

    :param file: path to yaml file to create
    :param data: dict of data to write in the file
    """
    stream = open(file, 'w')
    yaml.width = 4096
    yaml.dump(data, stream)