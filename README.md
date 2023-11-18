# Receipt Data Prediction
Receipt data prediction of coming months of the year 2022 given the receipt for the year 2021.

## Setting up. 
We have used PyTorch to write the model that can run on CPU machine and the code can be run locally. 

Clone the repository using

```
git clone git@github.com:gargsid/receipt_data_prediction_app.git
```

Then go into the directory `receipt_data_prediction_app` using

```
cd receipt_data_prediction_app
```

Next, install the requirements

```
pip install -r requirements.txt
```

## Training the models

It is recommended to train the models on your machine to support different hardware. Training only takes 1-2 minutes. To train the models, just run

```
python main.py
```

The above command with store the trained models in `assets/models` directory. 

## Running the Streamlit app

To test the model interactively in the browser use, 

```
streamlit run app.py
```