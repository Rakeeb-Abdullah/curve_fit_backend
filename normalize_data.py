import numpy as np

def normalize_data(x):
    mu=np.mean(x,axis=0)
    sigma=np.std(x,axis=0)
    x_=(x-mu)/sigma
    return x_

def return_original_wb(w,b,x):
    w_=w/np.std(x,axis=0)
    b_ = b- np.sum((w * np.mean(x,axis=0)) / np.std(x,axis=0))
    w_,b_=np.round(w_,2),np.round(b_,2)
    return w_,b_
