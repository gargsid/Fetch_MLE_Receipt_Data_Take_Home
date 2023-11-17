import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_utils import * 
from hyperparams import params 

def get_model(seq_len, hidden):
  model = nn.Sequential(
      nn.Linear(seq_len, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Linear(hidden, 1)
  )
  return model

def receipt_predictor_pipeline(seq_len, receipts):
    train_loader, test_loader = dataset_pipeline(seq_len, receipts)

    model = get_model(seq_len, params['hidden'])

    epochs = 10000
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-6)
    save_model_name = params['save_model_name'].format(seq_len)

    best_loss = 1e18
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.
        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float)
            labels = labels.to(torch.float)
            preds = model(inputs)
            loss = criterion(labels, preds.view(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        eval_loss = 0.

        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float)
            labels = labels.to(torch.float)
            preds = model(inputs)
            loss = criterion(labels, preds.view(-1))

        eval_loss = loss.item()
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), save_model_name)
        # if (epoch+1)%1000 == 0:
        #     print('epoch:', epoch+1, 'train:', train_loss, 'eval:', eval_loss)

    print('Evaluating model at least eval loss step....')
    model.load_state_dict(torch.load(save_model_name))
    model.eval()
    for inputs, labels in test_loader:
        inputs = inputs.to(torch.float)
        labels = labels.to(torch.float)
        preds = model(inputs)
    preds = preds.cpu().detach().numpy().squeeze()
    l = labels.cpu().detach().numpy().squeeze()
    error = ((preds - l)**2).mean()**0.5
    # print('test_labels:',[ll for ll in l])
    # print('preds:',[pred for pred in preds])
    print('rmse error:', error)

def get_best_model(receipts):
    best_rmse = 1e8
    best_seq_len = None

    errors = []

    for seq_len in params['sequence_lengths']:
        train_loader, test_loader = dataset_pipeline(seq_len, receipts)
        model = get_model(seq_len, params['hidden'])

        save_model_name = params['save_model_name'].format(seq_len)
        model.load_state_dict(torch.load(save_model_name))

        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float)
            labels = labels.to(torch.float)
            preds = model(inputs)

        preds = preds.detach().numpy().squeeze()
        l = labels.detach().numpy().squeeze()
        error = ((preds - l)**2).mean()**0.5
        print('seq_len:', seq_len, 'rmse error:', error)
        errors.append(error)

        if error < best_rmse:
            best_rmse = error
            best_seq_len = seq_len

    plt.plot(params['sequence_lengths'], errors)
    plt.xlabel('sequence_lengths')
    plt.ylabel('Validation RMSE values')
    plt.savefig('assets/val_rmse_values.png')
    plt.close()

    return best_seq_len

def prediction_pipeline(month_id, seq_len, receipts):
    model = get_model(seq_len, params['hidden'])
    save_model_name = params['save_model_name'].format(seq_len)
    model.load_state_dict(torch.load(save_model_name))
    x = receipts[-seq_len:]
    for _ in range(month_id):
        x = torch.from_numpy(np.array(x)).unsqueeze(0).to(torch.float)
        with torch.no_grad():
            pred = model(x)
        pred = pred.item()
        x = list(x[0].tolist())
        x = x[1:] + [pred]
        # print('month:', month+1, 'pred_receipts:', pred)

    return pred

