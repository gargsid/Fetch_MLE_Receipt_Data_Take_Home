from data_utils import * 
from receipt_prediction import *
from hyperparams import params 

import os 
os.makedirs('assets/data', exist_ok=True)
os.makedirs('assets/models', exist_ok=True)

df = read_csv()
receipts = list(df['Receipt_Count'].values)
# print(receipts)

for seq_len in params['sequence_lengths']:
    print('Training for Context Len:', seq_len)
    receipt_predictor_pipeline(seq_len, receipts)

best_seq_len = get_best_model(receipts)
print('best_seq_len:', best_seq_len)

for m in range(1,13):
    receipt_pred = prediction_pipeline(m, best_seq_len, receipts)
    print('month:', m, 'receipts:', receipt_pred)