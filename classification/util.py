import math


def mid_point(val_one, val_two):
    """
    :param val_one: lower bound
    :param val_two: upper bound
    :return: the mid point of two bounds
    """
    return (val_one*1.0 + val_two*1.0) / 2.0


def tree_log(val):
    """
    Customized log for building decision trees.
    :param val: The value to take the log of.
    :return: If val is 0, 0 is returned. Else, log2(val).
    """
    if val == 0:
        return 0
    else:
        return math.log2(val)
