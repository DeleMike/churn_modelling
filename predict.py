import pickle
import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_v0.bin'
print('loading model...')
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print('Loaded model: ',  (dv, model))

app = Flask(__name__,)

customer = {
   "geography": "france",
   "gender": "female",
   "hascrcard": "yes",
   "isactivemember": "no",
   "creditscore": 562,
   "age": 57,
   "tenure": 3,
   "balance": 0.0,
   "numofproducts": 3,
   "estimatedsalary": 6554.97
}

@app.route('/predict', methods=['POST'])
def predict():
   customer = request.get_json()
   X = dv.transform([customer])
   dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
   exited_prob  = model.predict(dtest)[0]
   churn = exited_prob >= 0.5
   results = {
      'exit_probabilty': float(exited_prob),
      'churn': bool(churn),
   }
   return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
