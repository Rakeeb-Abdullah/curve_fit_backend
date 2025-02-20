from flask import Flask,request,jsonify
from linear_regression import gradient_descent
import numpy as np
from normalize_data import normalize_data,return_original_wb
from flask_cors import cross_origin

app = Flask(__name__)

def graphing_data(x_max,x_min,w,b,m):
    x=np.linspace(x_min-(((x_max-x_min)/m)),x_max+((x_max-x_min)/m))
    y=x*w+b
    result=[{"x":xi,"y":yi} for xi,yi in zip(x,y)]
    return result

@app.route("/data/",methods=["GET","POST"])
@cross_origin(supports_credentials=True)
def hello_world():
    data=request.json
    X=np.array(data["X"]).reshape(-1,1)
    X_n = normalize_data(X)

    y=np.array(data["Y"])

    w,b,c=gradient_descent(X_n,y,data["i"],data["a"])
    w,b=return_original_wb(w,b,X)
    graph_codinates=graphing_data(X.max(),X.min(),w,b,X.shape[0])
    
    
    return jsonify({
        "m":w.tolist(),
        "c":b,
        "cost":c,
        "codinates":graph_codinates
    })

