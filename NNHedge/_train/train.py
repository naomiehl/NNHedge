import torch
from .._utils.utils import resolve_shape
import numpy as np

def single_epoch_train(model, optimizer, trainloader, loss_func, epoch, model_type:str, K=100):
    running_loss = 0.0 
    
    if model_type not in ['RNN','TCN','ATTENTION','SpanMLP','NET']:
        raise ValueError('Please use an available type of model. Available Models: RNN | TCN | ATTENTION |MLP or NET')

    model.train()
    for i, data in enumerate(trainloader):
        span, C, Sn, Sn_1 = data
        if model_type == 'RNN' or model_type =='TCN':
            span = torch.unsqueeze(span,-1)
        C    = resolve_shape(C)
        Sn   = resolve_shape(Sn)
        Sn_1 = resolve_shape(Sn_1)
        
         # Zero the parameter gradients
        optimizer.zero_grad()
        # z = 0
        # for i in range (ts-1):
            #z = z + model[i](span[i]) * (span[i+1] - span[i])
        # Forward + Backward + Optimize
        outputs = model(span) #uniquement un reseau de neurone
        #output = delta en 0
        # Sn - Sn-1 = increment du spot 
        # avoir un vect de S de la taille de discretisation de la maturite
        # outputs(de s0) * (s1-s0) + output( de s1) *(s2-s1) + ...
        # loss = loss_func(z - C (+/-) torch.max(span[ts-1] - K, torch.zeros(256, 1)), torch.zeros(256, 1))
        # construction de notre cout de hedge - Cout de friction - payoff
        loss = loss_func(outputs * (Sn - Sn_1) - C - torch.max(Sn - K, torch.zeros(256, 1)), torch.zeros(256, 1))
        loss.backward()
        optimizer.step()
        # un seul reseau de neurone qui prend le spot en 0, en 1 ..
        # soit on prend un model qui prend deux input le temps et le spot
         # Print statistics
        running_loss += loss.item()

    print('[%d] loss: %.6f' % (epoch + 1, running_loss))   

@torch.no_grad()
def single_epoch_test(model, testloader, model_type:str, K=100):
    y_val = []
    for i, data in enumerate(testloader):
        span, C, Sn, Sn_1 = data
        if model_type == 'RNN' or model_type =='TCN':
            span = torch.unsqueeze(span,-1)
        C    = resolve_shape(C)
        Sn   = resolve_shape(Sn)
        Sn_1 = resolve_shape(Sn_1)
        output = model(span)
        output = output * (Sn - Sn_1) - C + torch.max(Sn - K, torch.zeros(100, 1))

        y_val.extend(output.detach().tolist())
    return np.array(y_val).reshape(-1)
