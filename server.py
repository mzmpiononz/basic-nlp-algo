import pickle
from flask import Flask, request, jsonify
from utils import inference_preprocess

intent_list = ['reservation', 'colis', 'filan-kevitra']

app = Flask(__name__)
# remplacer par le model de votre choix
path_model = "transbot_model_6.pkl"
transbot_model = pickle.load(open(path_model, 'rb'))

@app.route('/', methods=['POST'])
def intent_pred():
    data = request.get_json()
    pred = intent_list[transbot_model.predict(inference_preprocess(data['phrase']))[0]]
    return jsonify(pred)
        
if __name__ == '__main__':
    app.run(debug=True, port='5000')
