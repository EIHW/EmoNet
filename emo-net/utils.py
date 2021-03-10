import numpy as np

def array_list_equal(a_list, b_list):
    if type(a_list) == list and type(b_list) == list:
        if len(a_list) != len(b_list):
            return False
        else:
            for a, b in zip(a_list, b_list):
                if not np.array_equal(a,b):
                    return False
            return True
    elif type(a_list) == np.array and type(b_list) == np.array:
        return np.array_equal(a_list, b_list)
    else:
        return False