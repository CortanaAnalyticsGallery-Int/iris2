import pickle
import sys
import os
import pandas as pd
import numpy as np
from azure.ml.api.schema.dataTypes import DataTypes
from azure.ml.api.schema.sampleDefinition import SampleDefinition
import azure.ml.api.realtime.services as amlo16n

def init():
    global clf2
    # load the model back from the 'outputs' folder into memory
    print("Import the model from model.pkl")
    f2 = open('./outputs/model.pkl', 'rb')
    clf2 = pickle.load(f2)

def run(npa):
    global clf2
    if isinstance(npa, str):
        print("convert string to array")
        finalarr = np.array(np.array(list(npa)))
    else:
        finalarr = npa
    print(finalarr.shape)
    pred = clf2.predict(npa)
    
    retdf = pd.DataFrame(data={"Scored Labels":np.squeeze(np.reshape(pred, newshape= [-1,1]), axis=1)})
    return str(retdf)

def main():
    init()
    # predict a new sample
    #X_new = [[3.0, 3.6, 1.3]]
    X_new1 = np.array([[ 5.1,  4.5,  1.4, 2.0], 
    [ 4.9,  3.0,  1.4, 0.2], 
    [ 4.7,  3.2,  1.3, 0.2],
    [ 5.6,  3.1,  1.5, 1.0],
    [ 6.3,  3.3,  6.0, 2.5],
    [ 5.0,  3.6,  1.4, 0.2]])
    print(run(X_new1))
    
    print("Calling prepare schema")
    inputs = {"npa": SampleDefinition(DataTypes.NUMPY, X_new1)}
    amlo16n.generate_schema(inputs=inputs,
                            filepath="outputs/schema.json",
                            run_func=run)
    amlo16n.generate_main(user_file="score.py", schema_file="outputs/schema.json",
                          main_file_name="outputs/main.py")
    print("End of prepare schema")
    
if __name__ == "__main__":
    main()