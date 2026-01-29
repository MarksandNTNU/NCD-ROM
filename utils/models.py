import torch
import gc
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from IPython.display import clear_output as clc
from .processdata import mse, mre, num2p
from torch.optim.lr_scheduler import StepLR
import torchcde

class SHRED(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, decoder_sizes = [350, 400], dropout = 0.1, activation = torch.nn.ReLU(), bidirectional = False):
        '''
        SHRED model definition
        
        
        Inputs
        	input size (e.g. number of sensors)
        	output size (e.g. full-order variable dimension)
        	size of LSTM hidden layers (default to 64)
        	number of LSTM hidden layers (default to 2)
        	list of decoder layers sizes (default to [350, 400])
        	dropout parameter (default to 0.1)
        '''
            
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers,
                                  bidirectional = bidirectional,
                                  batch_first=True)
        
        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, hidden_size)
        decoder_sizes.append(output_size)

        for i in range(len(decoder_sizes)-1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i != len(decoder_sizes)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(activation)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        
        # Move hidden states to the same device as the model parameters
        device = next(self.parameters()).device
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        _, (output, _) = self.lstm(x, (h_0, c_0))
        output = output[-1].view(-1, self.hidden_size)
    
        for layer in self.decoder:
            output = layer(output)

        return output

    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,  hidden_size = 64, hidden_layers = 2, dropout = 0.1, activation = torch.nn.ReLU()):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        layers = []
            
        layers.append(torch.nn.Linear(hidden_channels, hidden_size))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
            layers.append(torch.nn.Dropout(dropout))
        
        layers.append(torch.nn.Linear(hidden_size, hidden_channels*input_channels) )
        
        self.func = torch.nn.Sequential(*layers)    

    def forward(self, t, z):        
        z = self.func(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_size = 64, hidden_layers = 2, dropout = 0.1, activation = torch.nn.ReLU()):
        '''
        Neural CDE model definition
        
        
        Inputs
        	input size (e.g. number of sensors)
        	output size (e.g. full-order variable dimension)
        	size of CDE hidden layers (default to 64)
        	number of CDE hidden layers (default to 2)
        	list of decoder layers sizes (default to [350, 400])
        	dropout parameter (default to 0.1)
        '''
            
        super(NCDE,self).__init__()
    

        self.func = CDEFunc(input_channels, hidden_channels, hidden_layers=hidden_layers, hidden_size=hidden_size, dropout=dropout, activation=activation)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

    def forward(self, x):
        
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
        # Move hidden states to the same device as the model parameters
        device = next(self.parameters()).device
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z0 = z0.to(device)

        h_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)
        return h_T[:, -1] 
        # output = h_T
        # for layer in self.decoder:
        #     output = layer(output)

        # return output


def fit(model, train_dataset, valid_dataset, batch_size = 64, epochs = 400, optim = torch.optim.Adam, lr = 1e-3, loss_fun = mse, loss_output = mse, formatter = num2p, verbose = False, patience = 5, step_size=200, gamma = 1/2, device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"), collate_fn = None):
    '''
    Neural networks training
    
    Inputs
    	model (`torch.nn.Module`)
    	training dataset (`torch.Tensor`)
    	validation dataset (`torch.Tensor`)
    	batch size (default to 64)
    	number of epochs (default to 4000)
    	optimizer (default to `torch.optim.Adam`)
    	learning rate (default to 0.001)
        loss function (defalut to Mean Squared Error)
        loss value to print and return (default to Mean Relative Error)
        loss formatter for printing (default to percentage format)
    	verbose parameter (default to False) 
    	patience parameter (default to 5)
    '''

    # ========================================
    # DEVICE SETUP - Move all data to model's device at once
    # ========================================
    
    # Move training dataset to device
    if isinstance(train_dataset.X, (list, dict)):
        if isinstance(train_dataset.X, list):
            train_dataset.X = [tensor.to(device) for tensor in train_dataset.X]
        else:
            train_dataset.X = {key: tensor.to(device) for key, tensor in train_dataset.X.items()}
    else:
        train_dataset.X = train_dataset.X.to(device)
    train_dataset.Y = train_dataset.Y.to(device)
    
    # Move validation dataset to device  
    if isinstance(valid_dataset.X, (list, dict)):
        if isinstance(valid_dataset.X, list):
            valid_dataset.X = [tensor.to(device) for tensor in valid_dataset.X]
        else:
            valid_dataset.X = {key: tensor.to(device) for key, tensor in valid_dataset.X.items()}
    else:
        valid_dataset.X = valid_dataset.X.to(device)
    valid_dataset.Y = valid_dataset.Y.to(device)
    
    # ========================================
    # TRAINING SETUP
    # ========================================
    
    if collate_fn == None:
        train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    else:   
        train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, collate_fn = collate_fn)
    optimizer = optim(model.parameters(), lr = lr)

    scheduler = StepLR(optimizer, step_size, gamma=gamma)

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            print("Epoch:", epoch, "Batch:", k+1, "/", len(train_loader), end = "\r")
            def closure():
                outputs = model(data[0])
                optimizer.zero_grad()
                loss = loss_fun(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)
            

        model.eval()
        scheduler.step()

        with torch.no_grad():
            # compute scalar losses (move to CPU floats immediately to avoid keeping device tensors)
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            # store Python floats to avoid holding device tensors in the history lists
            train_error_list.append(float(train_error.detach().cpu().item()))
            valid_error_list.append(float(valid_error.detach().cpu().item()))
        
        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + formatter(train_error_list[-1]) + " \t Validation loss = " + formatter(valid_error_list[-1]) + " "*10 + " \t learning rate = " + str(scheduler.get_last_lr()[0]),  end = "\r")

        # Use scalar comparison on the stored float history
        if valid_error_list[-1] == min(valid_error_list):
            patience_counter = 0
            # store best params on CPU to avoid keeping GPU/MPS tensors around
            best_params = {k: v.cpu() for k, v in deepcopy(model.state_dict()).items()}
        else:
            patience_counter += 1

        if patience_counter == patience:
            # load best params (map to current device)
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in best_params.items()})

            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            
            if verbose == True:
                print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error) + " \t learning rate = " + str(scheduler.get_last_lr()[0]))
         
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    # restore best parameters (map to device)
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in best_params.items()})
    
    # Data is already on correct device - no need for device handling  
    train_error = loss_output(train_dataset.Y, model(train_dataset.X))
    valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
    
    if verbose == True:
        print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))

    # Move datasets back to CPU to avoid leaving large tensors on device
    try:
        if isinstance(train_dataset.X, (list, dict)):
            if isinstance(train_dataset.X, list):
                train_dataset.X = [t.cpu() for t in train_dataset.X]
            else:
                train_dataset.X = {k: t.cpu() for k, t in train_dataset.X.items()}
        else:
            train_dataset.X = train_dataset.X.cpu()
        train_dataset.Y = train_dataset.Y.cpu()

        if isinstance(valid_dataset.X, (list, dict)):
            if isinstance(valid_dataset.X, list):
                valid_dataset.X = [t.cpu() for t in valid_dataset.X]
            else:
                valid_dataset.X = {k: t.cpu() for k, t in valid_dataset.X.items()}
        else:
            valid_dataset.X = valid_dataset.X.cpu()
        valid_dataset.Y = valid_dataset.Y.cpu()
    except Exception:
        # Best-effort move back to CPU; if it fails, let caller handle dataset objects
        pass

    # Clean up large local objects and free device caches
    try:
        del optimizer
    except Exception:
        pass
    try:
        del train_loader
    except Exception:
        pass
    try:
        del scheduler
    except Exception:
        pass

    optimizer = None
    train_loader = None
    scheduler = None

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    # return numpy arrays built from Python float lists
    return np.array(train_error_list), np.array(valid_error_list)
 
def forecast(forecaster, input_data, steps, nsensors):
    '''
    Forecast time series in time
    Inputs
    	forecaster model (`torch.nn.Module`)
        starting time series of dimension (ntrajectories, lag, nsensors+nparams)
    	number of forecasting steps
        number of sensors
    Outputs
        forecast of the time series in time
    '''   

    forecast = []
    for i in range(steps):
        forecast.append(forecaster(input_data))
        temp = input_data.clone()
        input_data[:,:-1] = temp[:,1:]
        input_data[:,-1, :nsensors] = forecast[i]

    return torch.stack(forecast, 1)


