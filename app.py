import os
import base64
import io
import numpy as np
# Configure TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# Ensure TensorFlow uses CPU
tf.config.set_visible_devices([], 'GPU')
import cv2
from flask import Flask, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from PIL import Image

app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version='1.0',
    title='Plant Disease Detection API',
    description='A RESTful API for detecting plant diseases from uploaded images using a TensorFlow model. Upload an image to get disease classification, confidence, advice, and a base64-encoded image with detected edges.',
    doc='/swagger',
    contact='Yayah Habib Waritay',
    contact_email='yayah.waritay@njala.edu.sl',
    validate=True
)

ns = api.namespace('api/v1', description='Plant Disease Detection Operations')
MODEL_PATH = 'plant_model_v5-beta.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}. Please ensure 'plant_model_v5-beta.h5' is in the project directory.")
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_names = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Not a plant',
    5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew',
    7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy',
    12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight',
    23: 'Potato___healthy',
    24: 'Raspberry___healthy',
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch',
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}

# Advice for each disease
DISEASE_ADVICE = {
    'Apple___Apple_scab': 'Apply fungicides like captan or myclobutanil in early spring. Remove and destroy fallen leaves to reduce spore spread.',
    'Apple___Black_rot': 'Prune infected branches, apply fungicides like sulfur, and remove mummified fruit. Improve air circulation around trees.',
    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible. Apply fungicides like triadimefon during wet spring weather.',
    'Apple___healthy': 'Maintain regular watering and fertilization to keep your apple trees healthy.',
    'Blueberry___healthy': 'Continue good cultural practices like proper pruning and mulching to maintain blueberry health.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply sulfur-based fungicides and improve air circulation by pruning. Avoid overhead watering.',
    'Cherry_(including_sour)___healthy': 'Keep cherries healthy with consistent watering and annual pruning.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant varieties, apply fungicides like azoxystrobin, and rotate crops to reduce infection.',
    'Corn_(maize)___Common_rust_': 'Plant resistant hybrids and apply fungicides like mancozeb if rust appears early in the season.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant corn varieties, apply foliar fungicides, and practice crop rotation.',
    'Corn_(maize)___healthy': 'Ensure proper spacing and soil fertility to keep corn thriving.',
    'Grape___Black_rot': 'Apply fungicides like myclobutanil, remove infected berries, and prune for better air flow.',
    'Grape___Esca_(Black_Measles)': 'No cure; remove and destroy affected vines. Avoid wounding vines during pruning.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides like captan and improve canopy ventilation through pruning.',
    'Grape___healthy': 'Maintain vine health with proper irrigation and trellising.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure; remove and destroy infected trees. Control psyllid vectors with insecticides.',
    'Peach___Bacterial_spot': 'Use resistant varieties, apply copper-based bactericides, and avoid overhead irrigation.',
    'Peach___healthy': 'Keep peaches healthy with regular pruning and balanced fertilization.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper sprays, remove infected plant parts, and avoid working with wet plants.',
    'Pepper,_bell___healthy': 'Maintain pepper health with well-drained soil and adequate spacing.',
    'Potato___Early_blight': 'Apply fungicides like chlorothalonil, remove infected leaves, and practice crop rotation.',
    'Potato___Late_blight': 'Use resistant varieties, apply fungicides like mancozeb, and destroy infected tubers.',
    'Potato___healthy': 'Keep potatoes healthy with proper hilling and watering practices.',
    'Raspberry___healthy': 'Continue good pruning and mulching to maintain raspberry health.',
    'Soybean___healthy': 'PestDiagnosis proper soil drainage and crop rotation to keep soybeans healthy.',
    'Squash___Powdery_mildew': 'Apply fungicides like sulfur, improve air circulation, and avoid overhead watering.',
    'Strawberry___Leaf_scorch': 'Remove and destroy affected leaves, apply fungicides, and improve soil drainage.',
    'Strawberry___healthy': 'Keep strawberries healthy with mulching and regular watering.',
    'Tomato___Bacterial_spot': 'Use copper-based sprays, remove infected parts, and avoid overhead watering.',
    'Tomato___Early_blight': 'Apply fungicides like chlorothalonil, prune lower leaves, and mulch around plants.',
    'Tomato___Late_blight': 'Use resistant varieties, apply fungicides like copper, and destroy infected plants.',
    'Tomato___Leaf_Mold': 'Improve ventilation, apply fungicides like mancozeb, and avoid wetting foliage.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves, apply fungicides, and mulch to prevent soil splash.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides or insecticidal soap, increase humidity, and remove heavily infested leaves.',
    'Tomato___Target_Spot': 'Apply fungicides like azoxystrobin, remove infected leaves, and improve air circulation.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies with insecticides, use resistant varieties, and remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'Disinfect tools, remove infected plants, and avoid handling plants when wet.',
    'Tomato___healthy': 'Maintain tomato health with staking, watering at the base, and balanced fertilization.'
}

detect_parser = api.parser()
detect_parser.add_argument(
    'image',
    location='files',
    type=FileStorage,
    required=True,
    help='Plant image in JPG or PNG format (max 5MB)'
)

# Updated response model to match the desired format
detect_response = api.model('DetectResponse', {
    'status': fields.String(description='Response status', example='success', enum=['success', 'error']),
    'result': fields.Nested(api.model('Result', {
        'confidence': fields.Float(description='Overall prediction confidence score (0 to 100)', example=100.0),
        'detections': fields.List(
            fields.Nested(api.model('Detection', {
                'disease': fields.String(description='Predicted disease or healthy class', example='Potato___Late_blight'),
                'confidence': fields.Float(description='Prediction confidence score (0 to 100)', example=100.0),
                'advice': fields.String(description='Treatment advice for the detected disease', example='Use resistant varieties, apply fungicides like mancozeb, and destroy infected tubers.'),
            })),
            description='List of detected diseases with confidence and advice'
        ),
        'bounded_image': fields.String(description='Base64-encoded PNG image with red bounding box', example='data:image/png;base64,iVBOR...')
    }))
})

error_response = api.model('ErrorResponse', {
    'status': fields.String(description='Response status', example='error'),
    'message': fields.String(description='Error message', example='No image file provided')
})

health_response = api.model('HealthResponse', {
    'status': fields.String(description='API health status', example='healthy'),
    'version': fields.String(description='API version', example='1.0.0')
})

welcome_response = api.model('WelcomeResponse', {
    'message': fields.String(description='Welcome message', example='Welcome to the Plant Disease Detection API'),
    'endpoints': fields.List(fields.String, description='Available API endpoints', example=['/api/v1/health', '/api/v1/detect'])
})

def edge_and_cut(img, threshold1, threshold2):
    """Apply Canny edge detection and draw bounding box"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    emb_img = img.copy()
    edges = cv2.Canny(gray_img, threshold1, threshold2)
    edge_coors = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] != 0:
                edge_coors.append((i, j))

    if len(edge_coors) == 0:
        return emb_img

    row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
    row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
    col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
    col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]

    emb_color = np.array([255, 0, 0], dtype=np.uint8)
    emb_img[row_min-10:row_min+10, col_min:col_max] = emb_color
    emb_img[row_max-10:row_max+10, col_min:col_max] = emb_color
    emb_img[row_min:row_max, col_min-10:col_min+10] = emb_color
    emb_img[row_min:row_max, col_max-10:col_max+10] = emb_color

    return emb_img

def preprocess_and_predict(image):
    """Preprocess image and make prediction"""
    img_array = tf.image.resize(image, [256, 256])
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = tf.argmax(prediction[0], axis=-1)
    confidence = float(np.max(prediction[0]))
    return predicted_class.numpy(), confidence

@ns.route('/detect')
class Detect(Resource):
    @ns.doc('detect_plant_disease', description='Detect plant diseases from an uploaded image. Returns the predicted disease class, confidence score, treatment advice, and a base64-encoded image with a red bounding box around detected features.')
    @ns.expect(detect_parser)
    @ns.marshal_with(detect_response, code=200, description='Successful prediction')
    @ns.response(400, 'Bad Request', error_response)
    @ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Detect plant diseases from an uploaded image"""
        args = detect_parser.parse_args()
        image_file = args['image']
        if not image_file:
            return {'status': 'error', 'message': 'No image file provided'}, 400

        try:
            if image_file.mimetype not in ['image/jpeg', 'image/png']:
                return {'status': 'error', 'message': 'Invalid image format. Use JPG or PNG'}, 400

            image = Image.open(image_file).convert('RGB')
            img_array = np.array(image)
            predicted_class, confidence = preprocess_and_predict(img_array)
            confidence_percent = confidence * 100  # Convert to percentage

            if confidence < 0.60:
                return {
                    'status': 'success',
                    'result': {
                        'confidence': confidence_percent,
                        'detections': [{
                            'disease': 'Unknown',
                            'confidence': confidence_percent,
                            'advice': 'The image you uploaded might not be in the dataset. Try making your leaf background white.'
                        }],
                        'bounded_image': f'data:image/png;base64,{base64.b64encode(np.array(Image.fromarray(img_array)).tobytes()).decode("utf-8")}'
                    }
                }, 200

            class_name = class_names[predicted_class]
            advice = DISEASE_ADVICE.get(class_name, 'No specific treatment advice available. Consult a local agronomist.')
            bounded_image = edge_and_cut(img_array, 200, 400)
            bounded_pil = Image.fromarray(bounded_image)
            buffered = io.BytesIO()
            bounded_pil.save(buffered, format="PNG")
            bounded_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {
                'status': 'success',
                'result': {
                    'confidence': confidence_percent,
                    'detections': [{
                        'disease': class_name,
                        'confidence': confidence_percent,
                        'advice': advice
                    }],
                    'bounded_image': f'data:image/png;base64,{bounded_base64}'
                }
            }, 200

        except Exception as e:
            return {'status': 'error', 'message': f'Prediction failed: {str(e)}'}, 500

@ns.route('/health')
class Health(Resource):
    @ns.doc('health_check', description='Check the health status of the API. Returns the API status and version.')
    @ns.marshal_with(health_response, code=200, description='API is healthy')
    def get(self):
        """Check the health status of the API"""
        return {'status': 'healthy', 'version': '1.0.0'}, 200

@ns.route('/')
class Welcome(Resource):
    @ns.doc('welcome', description='Welcome endpoint providing an overview of the API and available endpoints.')
    @ns.marshal_with(welcome_response, code=200, description='Welcome message with endpoint list')
    def get(self):
        """Welcome to the Plant Disease Detection API"""
        return {
            'message': 'Welcome to the Plant Disease Detection API',
            'endpoints': ['/api/v1/health', '/api/v1/detect']
        }, 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
