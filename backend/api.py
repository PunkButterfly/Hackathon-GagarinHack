from fastapi import FastAPI, File, UploadFile
import uvicorn
import os

import aiofiles
from datetime import datetime

import clickhouse_connect

import yaml

from models.classifier import TorchClassifier

with open('./backend/CH_CONFIG.yaml', 'r')  as f:
    config = yaml.safe_load(f)

CH_HOST = config['HOST']
CH_PORT = config['PORT']
CH_USERNAME = config['USER']
CH_PASSWORD = config['PASSWORD']

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
TMP_DIR = './backend/tmp_files/'

PORT = 8502

app = FastAPI()

def save_to_db(binary_data, file):
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, username=CH_USERNAME, password=CH_PASSWORD)

    s_bin = str(binary_data)[2:-1]

    client.command("INSERT INTO GagarinHack2024.queries (bindata, filename) VALUES ('{}', '{}')".format(s_bin, file.filename))

def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.get("/")
def root():
    return "Саша, мне вообще-то обидно, когда не доверяют моим данным."

@app.post("/process_image/")
async def process_image(file: UploadFile):
    if not file:
        return {"message": "No upload file sent"}
    elif not allowed_img(file.filename):
        return {"message": "Not allowed file extension"}
    else:
        out_file_name = f'tmp_{datetime.now()}_' + file.filename
        out_file_path = os.path.join(TMP_DIR, out_file_name)

        async with  aiofiles.open(out_file_path, 'wb') as out_file:
            binary_data = await file.read()  # async read
            await out_file.write(binary_data)  #

        # save_to_db(binary_data, file)

        model = TorchClassifier('./backend/models/weights/v1_weights.pt')

        img_result = model.process_img(out_file_path)
        print(img_result)

        return img_result

        #заглушка
        model_res = {
            "image_classes_probs": {'one': 0.33, 'two': 0.33, 'three': 0.33},
            "image_content": {'page_number': 1, 'other_content': 'ya lublu sobak'}
        }
        return model_res
        
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)