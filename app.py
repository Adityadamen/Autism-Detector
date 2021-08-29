from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('modelA.pkl', 'rb'))
model1 = pickle.load(open('adult.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/practice")
def practice():
    """ return the rendered template """
    return render_template("practice.html")
@app.route("/practice1")
def practice1():
    """ return the rendered template """
    return render_template("practice1.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [ x for x in request.form.values()]
    l=[]

    for i in range(len(int_features)-1):
        l.append(int_features[i+1])
    l = [int(i) for i in l]
    final_features = [np.array(l)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
        output='YES'
    else:
        output = 'NO'

    return render_template('practice.html', prediction_text='DO I HAVE AUTISM ? {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/predict1',methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [ x for x in request.form.values()]
    l=[]

    for i in range(len(int_features)-1):
        l.append(int_features[i+1])
    l = [int(i) for i in l]
    final_features = [np.array(l)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
        output='YES'
    else:
        output = 'NO'

    return render_template('practice1.html', prediction_text='DO I HAVE AUTISM ? {}'.format(output))


@app.route('/predict1_api',methods=['POST'])
def predict1_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model1.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run()
