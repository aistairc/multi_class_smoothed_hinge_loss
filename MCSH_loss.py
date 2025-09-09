import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

def huber(x, a=0.3, delta=1):
    '''
     y = g * x + b
     g = 2*a*delta

     g*de://us04web.zoom.us/j/74689769110?pwd=gb0yHdHWqcJOhsVTIBYvGALA9-Y0cQ.1ta + b = a * delta^2
     b = a*delta^2-g*delta
    '''
    g = 2*a*delta
    b = a*delta*delta-g*delta

    x1 = x.abs()
    x2 = x**2
    m = (x1.detach()<delta).to(dtype=torch.float)

    return m*(x2*a) + (1-m)*(x1*g+b)

class _MultiClassMarginLoss(nn.Module):
    def __init__(self, margin=1, true_class_score=1, adaptive=False ):
        super(_MultiClassMarginLoss, self).__init__()
        self.margin=margin
        self.true_class_score=true_class_score
        self.adaptive=adaptive

    def _func(self, x ):
        return x

    def forward( self, score, label ):
        x = score

        dim = [0,]+list(range(2,x.ndim))

        oh = F.one_hot( label, x.shape[1])

        xt = (x * oh).sum(dim=1,keepdim=True)

        x = x - xt + self.margin

        x = self._func(x)
        x = torch.clamp(x, max=88, out=None) 
        x = x.exp()

        x = x*(1-oh)

        x = torch.log( 2-x.shape[1] + x.sum(dim=1) )

        if( self.adaptive ):
            x = x.sum() / x.detach().count_nonzero().clip(min=1)
            if( x < 1 ):
                x = (x+self.eps) / (x.detach()+self.eps)
        else:
            x = x.mean()

        return x


class MultiClassSmoothedHingeLoss(_MultiClassMarginLoss):
    def __init__(self, margin=1, true_class_score=1, adaptive=False ):
        super(MultiClassSmoothedHingeLoss, self).__init__(margin, true_class_score, adaptive )

    def _func(self,x):
        return huber( F.relu(x), delta = self.margin)

