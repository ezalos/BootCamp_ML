# import sys
# import os
# path = os.path.join(os.path.dirname(__file__), '..', 'ex04')
# sys.path.insert(1, path)
# from prediction import predict_
# path = os.path.join(os.path.dirname(__file__), '..', '..', 'day00', 'ex03')
# sys.path.insert(1, path)


import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept
from prediction import predict_
