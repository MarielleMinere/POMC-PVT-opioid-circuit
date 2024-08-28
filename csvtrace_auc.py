import numpy as np
import pandas as pd
from sklearn.metrics import auc

#insert location of csv file containing data for a single recording
CSV_FILE = r''
START = 12000       # first frame for the AUC period
END = 24000         # last frame for the AUC period


if '__main__' == __name__:
    data = pd.read_csv(CSV_FILE, delimiter=';')
    array = data.squeeze()
    timeperiod = np.arange(START, END)
    array_auc = array[START:END]
    auc = auc(timeperiod, array_auc)
    print(auc)

