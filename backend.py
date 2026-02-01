from base64 import encodebytes
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
@app.after_request
def add_no_cache_headers(response):
    if request.path.startswith('/') and not request.path.startswith('/predict') and not request.path.startswith('/chat'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response
models = {}
encoders = {}
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return encoder.transform([encoder.classes_[0]])[0]
def train_models():
    print(" [System] Training AI Models... Please wait.")
    file_soil = 'Smart_Agriculture_10000_Rows_Final_No_Unknown.xlsx'
    if os.path.exists(file_soil):
        df = pd.read_excel(file_soil)
        df.columns = df.columns.str.strip()
        encs = {}
        for col in ['State', 'Soil_Type', 'Crop', 'Fertilizer']:
            le = LabelEncoder()
            df[f'{col}_En'] = le.fit_transform(df[col])
            encs[col] = le
        encoders['case1'] = encodebytes
        features = ['State_En', 'Soil_Type_En', 'Nitrogen', 'Phosphorus', 
                    'Potassium', 'pH', 'Temperature', 'Rainfall', 'Humidity']
        
        X = df[features]
        m_crop = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df['Crop_En'])
        X_full = X.copy()
        X_full['Crop_En'] = df['Crop_En']
        
        m_yield = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_full, df['Yield'])
        m_fert = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_full, df['Fertilizer_En'])
        
        models['case1'] = {'crop': m_crop, 'yield': m_yield, 'fert': m_fert}
        print(f" Case 1 (Soil) Ready - Trained on {len(df)} rows from {file_soil}")
    else:
        print(f" Error: {file_soil} not found.")
    file_hist = 'Smart_Agriculture_10000_Rows_Historical.xlsx'
    
    if os.path.exists(file_hist):
        df = pd.read_excel(file_hist)
        df.columns = df.columns.str.strip()
        
        encs = {}
        for col in ['State', 'Soil_Type', 'Previous_Crop', 'Recommended_Crop']:
            le = LabelEncoder()
            df[f'{col}_En'] = le.fit_transform(df[col])
            encs[col] = le
        encoders['case2'] = encs
        X = df[['State_En', 'Soil_Type_En', 'Previous_Crop_En']]
        m_crop = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df['Recommended_Crop_En'])
        models['case2'] = {'crop': m_crop}
        print(f" Case 2 (History) Ready - Trained on {len(df)} rows from {file_hist}")
    else:
        print(f" Error: {file_hist} not found.")
@app.route('/')
def home():
    response = app.send_static_file('index.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS) with no-cache headers"""
    if filename in ['style.css', 'script.js']:
        response = app.send_static_file(filename)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return app.send_static_file(filename)

@app.route('/predict/case1', methods=['POST'])
def predict_case1():
    try:
        if 'case1' not in models or 'case1' not in encoders:
            return jsonify({"error": "Model not trained. Please restart the server."}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        enc = encoders['case1']
        state_en = safe_transform(enc['State'], data.get('State', ''))
        soil_en = safe_transform(enc['Soil_Type'], data.get('Soil_Type', ''))
        inputs = [
            state_en, soil_en,
            float(data.get('Nitrogen', 0)),
            float(data.get('Phosphorus', 0)),
            float(data.get('Potassium', 0)),
            float(data.get('pH', 7.0)),
            float(data.get('Temperature', 25.0)),
            float(data.get('Rainfall', 1000.0)),
            float(data.get('Humidity', 50.0))
        ]
        crop_en = models['case1']['crop'].predict([inputs])[0]
        crop_name = enc['Crop'].inverse_transform([crop_en])[0]
        full_inputs = inputs + [crop_en]
        yield_val = models['case1']['yield'].predict([full_inputs])[0]
        fert_en = models['case1']['fert'].predict([full_inputs])[0]
        fert_name = enc['Fertilizer'].inverse_transform([fert_en])[0]
        
        return jsonify({
            "crop": crop_name,
            "yield": f"{yield_val:.2f} Quintals/Acre",
            "fertilizer": fert_name
        })
    except KeyError as e:
        print(f"Error in Case 1 - Missing key: {e}")
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except ValueError as e:
        print(f"Error in Case 1 - Invalid value: {e}")
        return jsonify({"error": "Invalid input value. Please check your numbers."}), 400
    except Exception as e:
        print(f"Error in Case 1: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction Failed. Check Input Values."}), 400

@app.route('/predict/case2', methods=['POST'])
def predict_case2():
    try:
        if 'case2' not in models or 'case2' not in encoders:
            return jsonify({"error": "Model not trained. Please restart the server."}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        enc = encoders['case2']
        
        inputs = [
            safe_transform(enc['State'], data.get('State', '')),
            safe_transform(enc['Soil_Type'], data.get('Soil_Type', '')),
            safe_transform(enc['Previous_Crop'], data.get('Previous_Crop', ''))
        ]
        
        crop_en = models['case2']['crop'].predict([inputs])[0]
        crop_name = enc['Recommended_Crop'].inverse_transform([crop_en])[0]
        
        return jsonify({
            "crop": crop_name,
            "yield": "N/A (Historical Data)",
            "fertilizer": "Standard NPK"
        })
    except KeyError as e:
        print(f"Error in Case 2 - Missing key: {e}")
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        print(f"Error in Case 2: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').lower()
    reply = "I am an AI. Ask me about Rice, Wheat, or Soil."
    
    if 'hello' in msg: reply = "Hello! How can I help your farm today?"
    elif 'rice' in msg: reply = "Rice needs clayey soil and high rainfall (~1200mm). Best fertilizer: Urea + DAP."
    elif 'wheat' in msg: reply = "Wheat needs loamy soil and cool temperatures (15-20°C). Best fertilizer: NPK 12-32-16."
    elif 'soil' in msg: reply = "Soil testing helps determine the exact NPK values needed for your farm."
    elif 'fertilizer' in msg: reply = "Urea provides Nitrogen. DAP provides Phosphorus. Potash provides Potassium."
    
    return jsonify({"reply": reply})

if __name__ == '__main__':
    train_models()
    app.run(port=5000, debug=True)