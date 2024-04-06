from flask import Flask, request

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

PORT = 8502

app = Flask(__name__)

def save_to_db(db=None, image=None, proceed_res= None):
    pass

def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_image', methods=['POST', 'GET'])
def process_image():
     
     if request.method == 'GET':
        #заглушка
        model_res = {
            "image_classes_probs": {'one': 0.33, 'two': 0.33, 'three': 0.33},
            "image_content": {'page_number': 1, 'other_content': 'ya lublu sobak'}
        }

        return model_res
     
     if request.method == 'POST':
        if 'image' not in request.files:
            return {'msg': 'no image'}
        
        image = request.files['image']

        if image not in allowed_img(image) :
            return {'msg': 'image extension is not allowed'} 
        
        # model_res = ...

        return model_res
     

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
