import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *

X1, y1 = get_feature_matrix(data='hod', method='concatenate')
X2, y2 = get_feature_matrix(data='hod', method='subtract')
X3, y3 = get_feature_matrix(data='gsc', method='concatenate')
X4, y4 = get_feature_matrix(data='gsc', method='subtract')
