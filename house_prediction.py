from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

house_data = {
    "area": [800,900,1000,1100,1200,1300,1500,1700,2000,2200,2500,2800,3000],
    "bedrooms": [1,2,2,2,3,3,3,3,4,4,4,5,5],
    "bathrooms": [1,1,2,2,2,2,3,3,3,4,4,4,5],
    "price": [30,35,40,45,50,55,65,75,90,110,130,150,170]
}

df = pd.DataFrame(house_data)

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
<title>House Price Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<style>
body{
    margin:0;
    font-family:'Poppins',sans-serif;
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#667eea,#764ba2);
}
.container{
    background:#fff;
    width:400px;
    padding:30px;
    border-radius:16px;
    box-shadow:0 20px 40px rgba(0,0,0,0.2);
}
.container h2{
    text-align:center;
    margin-bottom:5px;
}
.container p{
    text-align:center;
    color:#666;
    margin-bottom:20px;
}
label{
    font-size:14px;
    font-weight:500;
}
input{
    width:100%;
    padding:10px;
    margin:8px 0 16px 0;
    border-radius:8px;
    border:1px solid #ccc;
    outline:none;
}
input:focus{
    border-color:#667eea;
}
button{
    width:100%;
    padding:12px;
    border:none;
    border-radius:10px;
    background:#667eea;
    color:#fff;
    font-size:16px;
    font-weight:600;
    cursor:pointer;
}
button:hover{
    background:#5a67d8;
}
#result{
    margin-top:20px;
    text-align:center;
    font-size:18px;
    font-weight:600;
    color:#2d3748;
}
</style>
</head>
<body>
<div class="container">
    <h2>🏠 House Price Prediction</h2>
    <p>Major Data Science Project</p>

    <label>Area (sqft)</label>
    <input id="area" placeholder="e.g. 2000">

    <label>Bedrooms</label>
    <input id="bed" placeholder="e.g. 3">

    <label>Bathrooms</label>
    <input id="bath" placeholder="e.g. 2">

    <button onclick="predict()">Predict Price</button>
    <div id="result"></div>
</div>
<script>
function predict(){
fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({area:area.value,bed:bed.value,bath:bath.value})})
.then(res=>res.json())
.then(d=>result.innerHTML='Estimated Price: ₹ '+d.price+' Lakhs')
}
</script>
</body>
</html>
'''

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    price = model.predict([[float(d['area']), int(d['bed']), int(d['bath'])]])
    return jsonify({"price": round(price[0],2)})

if __name__ == '__main__':
    app.run(debug=True)
