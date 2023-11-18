# Receipt Data Prediction
Receipt data prediction of coming months of the year 2022 given the receipt for the year 2021. To follow the step-by-step walkthrough it is recommended to follow this [colab notebook](https://colab.research.google.com/drive/1ZFvFYzbbaIR9hoh6mW5QiG083GzhylmW?usp=sharing). It shows some data analysis and detailed descriptions of different functions. For only running the inference, please follow this brief [inference notebook](https://colab.research.google.com/drive/1jOHlJhDT6O6UBfSWkirRzZytYCTi6k1C?usp=sharing)

## Setting up. 
We have used PyTorch to write the model that can run on CPU machine and the code can be run locally. 

Clone the repository using

```
git clone https://github.com/gargsid/receipt_data_prediction_app.git
```

Then go into `receipt_data_prediction_app` directory using

```
cd receipt_data_prediction_app
```

Next, install the requirements

```
pip install -r requirements.txt
```

If you face any issues with running the code, you can also try replicating the environment using

```
conda env create -f environment.yml
conda activate env
```

The name of the created environment is `env`, which you can change by editing the environment.yml file.

## Running the Streamlit app

To test the model interactively in the browser use, 

```
streamlit run app.py
```

## Training the models

If you want to train the models you can use the following command. Training only takes 3-4 minutes. To train the models, justrun

```
python main.py
```

The above command with store the trained models in `assets/models` directory. 