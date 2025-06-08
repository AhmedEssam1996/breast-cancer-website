from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# تحميل النموذج
model = pickle.load(open('model.pkl', 'rb'))

# أسماء الخصائص الـ31 (من غير id و diagnosis)
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = []
        for feature in features:
            value = request.form.get(feature)
            input_features.append(float(value))
        
        prediction = model.predict([input_features])
        result = "خبيث (Malignant - M)" if prediction[0] == 1 else "حميد (Benign - B)"
        return render_template('index.html', prediction_text=f'النتيجة: {result}', features=features)
    except Exception as e:
        return f"حدث خطأ: {e}"

if __name__ == '__main__':
    app.run(debug=True)


