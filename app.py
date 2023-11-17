import streamlit as st 
from hyperparams import params 
from receipt_prediction import * 

df = read_csv()
receipts = list(df['Receipt_Count'].values)

month_to_id = {
    'Jan' : 1, 
    'Feb' : 2,
    'Mar' : 3,
    'Apr' : 4, 
    'May' : 5,
    'Jun' : 6,
    'Jul' : 7,
    'Aug' : 8,
    'Sep' : 9,
    'Oct' : 10,
    'Nov' : 11,
    'Dec' : 12,
}

# st.session_state.seq_len = 2
def best_model_predictions():
    # month: Jan,..,Dec
    # year: 2022, 2023 
    month = st.session_state.month_selected
    year = st.session_state.year_selected
    best_seq_len = st.session_state.seq_len
    month_id = month_to_id[month] 
    n_preds = 12 * (year - 2022) + month_id 
    
    model = get_model(best_seq_len, params['hidden'])
    save_model_name = params['save_model_name'].format(best_seq_len)
    model.load_state_dict(torch.load(save_model_name))
    
    x = receipts[-best_seq_len:]
    for _ in range(n_preds):
        x = torch.from_numpy(np.array(x)).unsqueeze(0).to(torch.float)
        with torch.no_grad():
            pred = model(x)
        pred = pred.item()
        x = list(x[0].tolist())
        x = x[1:] + [pred]
        # print('month:', month+1, 'pred_receipts:', pred)
    with prediction_col:
        prediction_box.text_input(label='Predicted Receipts', value=int(pred))
    return pred

def seq_preds(n_preds):
    seq_len = st.session_state.seq_len
    model = get_model(seq_len, params['hidden'])
    save_model_name = params['save_model_name'].format(seq_len)
    model.load_state_dict(torch.load(save_model_name))
    
    outputs = []
    x = receipts[-seq_len:]
    for _ in range(n_preds):
        x = torch.from_numpy(np.array(x)).unsqueeze(0).to(torch.float)
        with torch.no_grad():
            pred = model(x)
        pred = pred.item()
        x = list(x[0].tolist())
        x = x[1:] + [pred]
        outputs.append(pred)

    return outputs

st.title('Receipt Prediction App')

st.write('Welcome to the receipt prediction app!')

st.header('Overview', divider=True)

st.write(' We had receipt data for each month in 2021 and we trained a single \
         layer MLP model to predict the receipt data for the year 2022. The MLP is trained \
         to take sequences of receipts as input upto certain months and predicts the receipts \
         for the next month. The model makes predictions in a sequential manner in which \
         it adds the predicted number of receipts at the end of the sequence to make predictions \
         for the successive months (like RNN/LSTM or GPT-like models--transformer decoders) \
         Note that the certainty of the model reduces as the model make predictions too far \
         into the future. Therefore, we allow the predictions for only the years 2022, and 2023.')

st.header('Receipt Predictions With Best Performing Model', divider=True)

st.write('We trained multiple neural networks that takes context lengths of multiple size as inputs\
         like 2,3,...,9 and use this context to make predictions.')

st.write('Note: Model trained for context length of 2 gave the best results.')

month_col, year_col, slider_col = st.columns(3)
with month_col:
    st.session_state.month_selected = st.selectbox('Month', list(month_to_id.keys()), on_change=best_model_predictions)
with year_col:
    st.session_state.year_selected = st.selectbox('Year', [2022, 2023], on_change=best_model_predictions)
with slider_col:
    st.slider('Context Length', 2, 9, on_change=best_model_predictions, key='seq_len')

button_col, prediction_col, buff = st.columns(3)
with prediction_col:
    prediction_box = st.empty()
    prediction_box.text_input(label='Predicted Receipts')

with button_col:
    st.write('')
    predict_button = st.button('Make Predictions')    
    if predict_button:
        with prediction_col:
            predicted_receipts = best_model_predictions()
            # prediction_box.text_area(label='Predicted Receipts', value=int(predicted_receipts))

st.write('We can also generate a sequence of predictions starting from Jan 2022. Use the \#Preds slider\
         It will be use the Context Length slider value to make predictions.')

sl_col, text_col = st.columns(2)

with text_col:
    seq_box = st.empty()
    seq_box.text_area('Predicted Sequence', height=100) 

with sl_col:
    st.slider('#Preds', 1, 12, key='n_preds')
    generate_button = st.button('Generate Sequence')    
    if generate_button:
        outputs = seq_preds(st.session_state.n_preds)
        pred_seq = ''
        for i, x in enumerate(outputs):
            pred_seq += f'Pred #{i+1}: {int(x)}\n'
        # pred_seq = '\n'.join([str(x) for x in outputs])
        with text_col:
            seq_box.text_area('Predicted Sequence', pred_seq)
# st.session_state
