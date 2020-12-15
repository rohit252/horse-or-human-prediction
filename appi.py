# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:13:25 2020

@author: Home
"""

from flask import Flask, render_template, request
import os

import numpy as np
from keras.preprocessing import image
from keras.models import load_model



path = r'C:\Users\Home\Desktop\data\horse\k.h5'


model = load_model(path)

def pred(images):
    test_image = image.load_img(images,target_size=(150,150))
    test_image = image.img_to_array(test_image)/255
    test_image = np.expand_dims(test_image,axis=0)
    
    #result = model1.predict(test_image)
    
    result1 = model.predict_classes(test_image)
    
    
    if result1 == 0:
        return ('horse')
    else:
        return ('human')
        

# craete Flask instance
app = Flask(__name__,template_folder='template')

# render index.html page
@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/predict", methods = ['GET', 'POST'])

def predict():
    if request.method == 'POST':
        file = request.files['image'] # feed input
        filename = file.filename
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join(r'static\user uploaded', filename)
        file.save(file_path)
        
        print("@@ predcitiing class.........")
        pred1 = pred(images = file_path)
        
        return render_template('predict.html',pred_output=pred1,user_image=file_path)
    
if __name__ == "__main__":
    app.run(threaded=False)






