import numpy as np
import pandas as pd
import math
import torch
from torch import nn, Tensor
from torchtuples import Model
import torchtuples as tt
import scipy.stats

def gen_mask(row, col, percent=0.5, num_zeros=None):
    #### mask network 
    # adapted from 'https://blog.csdn.net/Kuo_Jun_Lin/article/details/115552545'
    if num_zeros is None:
        # Total number being masked is 0.5 by default.
        num_zeros = int(np.random.binomial(row * col,percent))#int((row * col) * percent)

    mask = np.hstack([
    	np.zeros(num_zeros),
        np.ones(row * col - num_zeros)])

    np.random.shuffle(mask)
    return mask.reshape(row, col)

class LinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask

        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask

class CustomizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=None):
        """
        Argumens
        ------------------
        mask [numpy.array]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute.
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(
            	torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Initialize the above parameters (weight & bias).
        self.init_params()

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
            # print('\n[!] CustomizedLinear: \n', self.weight.data.t())
        else:
            self.register_parameter('mask', None)

    def init_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(
        	input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None)


# net structure for DPLQR with mask
class dqNetSparse(nn.Module):

    def __init__(self,dim_nonpar,nodes,sparseRatio,bias=True,dropout=1e-5, 
                 activation=nn.ReLU, w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        '''
        sparseRatio: percentage of number of zero
        '''
        self.num_hid_layer = nodes[0]
        self.width = nodes[1] if nodes[1]>0 else None
            
        self.nonparLinear1 = nn.Linear(dim_nonpar,self.width,bias)
        self.nonparLinearList = nn.ModuleList([CustomizedLinear(self.width, self.width, 
                                              mask=gen_mask(self.width, self.width, sparseRatio))
                                              for _ in range(nodes[0])])
        self.nonparLinearEnd = nn.Linear(self.width,1,bias)
        
        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):#in_nonpar
        
        x = self.activation(self.nonparLinear1(x))#in_nonpar
        x = self.dropout(x)
        
        for f in self.nonparLinearList:
            x = self.activation(f(x))
            x = self.dropout(x)
            
        x = self.nonparLinearEnd(x)#out_nonpar
         
        return x#out_nonpar
    
    #def predict(self, in_nonpar):
        #return self.forward(in_nonpar)

## loss function
def check_loss(y_pred: Tensor, target: Tensor, tau: float) -> Tensor:
    errors = target-y_pred 
    u = torch.max(tau*errors,(tau-1)*errors)
    return u.mean()

class checkLoss(torch.nn.Module):
    
    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau
        
    def forward(self, g: Tensor, y: Tensor) -> Tensor:
        return check_loss(g,y,self.tau)  
