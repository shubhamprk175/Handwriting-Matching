from logistic_regression import *
from linear_regression import *
from neural_network import *
from preprocessing import *

X1, y1 = get_feature_matrix(data='hod', method='concatenate')
# X2, y2 = get_feature_matrix(data='hod', method='subtract')
# X3, y3 = get_feature_matrix(data='gsc', method='concatenate')
# X4, y4 = get_feature_matrix(data='gsc', method='subtract')

# logistic_regression(X1, y1)
# logistic_regression(X2, y2)
# logistic_regression(X3, y3)
# logistic_regression(X4, y4)



# linear_regression(X1, y1)
# linear_regression(X2, y2)
# linear_regression(X3, y3)
# linear_regression(X4, y4)



neural_network(X1, y1)
# neural_network(X2, y2)
# neural_network(X3, y3)
# neural_network(X4, y4)
