#                    EmoNet
# ==============================================================================
# Copyright (C) 2021 Maurice Gerczuk, Shahin Amiriparian,
# Sandra Ottl, Björn Schuller: University of Augsburg. All Rights Reserved.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
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