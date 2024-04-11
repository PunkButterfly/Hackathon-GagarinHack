from fastapi import FastAPI
import uvicorn

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

PORT = 8502
app = FastAPI()

app = FastAPI()

def save_to_db(db=None, image=None, proceed_res= None):
    pass

def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.get("/")
def root():
    return "Саша, мне вообще-то обидно, когда не доверяют моим данным."

@app.get('/process_image')
def process_image():
        #заглушка
        model_res = {
            "image_classes_probs": {'one': 0.33, 'two': 0.33, 'three': 0.33},
            "image_content": {'page_number': 1, 'other_content': 'ya lublu sobak'}
        }
        return model_res
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)