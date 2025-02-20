import matplotlib.pyplot as plt
import numpy as np



def visualize(X,y,w,b):
    m,n=X.shape

    x_range = np.linspace(-5, 6, 100)
    for j in range(n):
        plt.subplot(1,n,j+1)

        y_line = w[j] * x_range[0] + b

        plt.plot(x_range, y_line, color='red', label=f'y = {w}x + {b}')

        plt.scatter(X[:,j],y,marker='x',c='r')

    plt.show()





def predict(w,b,X):
    p=np.dot(w,X)+b
    return p

def compute_cost(X,y,w,b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        p=predict(w,b,X[i])
        cost+=(p-y[i])**2
    return cost/(2*m)

def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_db=0.
    dj_dw=np.zeros((n,))
    for i in range(m):
        err=predict(w,b,X[i])-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db+=err
    
    return dj_dw/m,dj_db/m

def gradient_descent(X,y,iterations,a):
    
    b = 0
    w = np.zeros(X.shape[1])
    c_l=[]
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(X,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db

        if i%1000==0:
            c=compute_cost(X,y,w,b)
            c_l.append(c)
            # print(compute_cost(X,y,w,b))
    # visualize(X,y,w,b)
    return w,b,np.round(c_l[-1],2)

