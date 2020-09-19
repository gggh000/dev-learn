import numpy as np
from basic_function_helper import do_square_using_c
#my_list = np.arange(1000)
my_list = range(0, 10)
squared_list = do_square_using_c(*my_list)
