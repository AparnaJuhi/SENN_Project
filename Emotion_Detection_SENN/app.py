from flask import Flask,render_template,request
import tensorflow as tf
from nltk.tokenize import word_tokenize
import numpy as np
app = Flask(__name__)
senn_model = tf.keras.models.load_model('sennModel.h5')
word_to_emb={}

def padd(arr):
    for i in range(100-len(arr)):
        arr.append('<pre>')
    return arr[:100]

def load_glove():
    with open('glove.6B.50d.txt',encoding="utf-8") as vocab_f:
        for line in vocab_f:
            word_to_emb[line.split()[0]]=[float(i) for i in line.split()[1:]] 
    word_to_emb['<pre>']=[0]*50

def prepare_data(s):
    s = word_tokenize(s)
    s = padd(s)
    arr = [[]]
    for word in s:
        if word in word_to_emb:
            arr[0].append(word_to_emb[word])
        else:
            arr[0].append([0]*50)
    return arr
        

@app.route('/')
def main():
    load_glove()
    return render_template('index.html')
        

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['tex']
    text = text.lower()
    arr = prepare_data(text)
    arr=np.array(arr)
    X = senn_model.predict([arr,arr])
    res = X.argmax()
    return render_template('predict.html',data=res)
    
if __name__ == '__main__':
    app.run(debug=True)