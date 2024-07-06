from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, Response
import uvicorn
import os
import io
import aiofiles
from datetime import datetime

from PIL import Image

from utils import *
from models.Pipeline import Pipeline

# workdir = './backend/'
workdir = '' # for docker

PORT = os.getenv('BACKEND_PORT')

TMP_DIR = f'{workdir}tmp_files/'
WEIGHTS_DIR = f'{workdir}models/weights/'

MOBILE_LAST_IMG_FILENAME = 'LATEST_MOBILE.jpg'

app = FastAPI()

pipeline = Pipeline(WEIGHTS_DIR, TMP_DIR)
 
@app.get("/")
def root():
    return "Avialiable"


async def save_image(file_binary, filename= None):
    if filename is None:
        filename = f'tmp_{datetime.now()}_.jpg'
    img_file_path = os.path.join(TMP_DIR, filename)

    print(img_file_path)

    async with aiofiles.open(img_file_path, 'wb') as out_file:
        await out_file.write(file_binary) 

    return img_file_path, file_binary


@app.post("/detect/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
    
        img_file_path, binary_img_data = await save_image(file)

        classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        save_to_db(binary_img_data, img_file_path)

        # pipeline = Pipeline(WEIGHTS_DIR, TMP_DIR, classifier_weights_name = 'weights_71_0.94_0.93_0.93_0.93.pt')
        # если хотим заюзать веса большого датасета

        response = format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path)

        predict_img = Image.open(predict_img_path)
        bytes_image = io.BytesIO()

        predict_img.save(bytes_image, format='PNG')
        
        return Response(content=bytes_image.getvalue(), headers=response, media_type="image/png")
    
@app.post("/detect_for_mobile/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
        img_file_path, binary_img_data = await save_image(file, filename=MOBILE_LAST_IMG_FILENAME)

        classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        save_to_db(binary_img_data, img_file_path)

        Image.open(predict_img_path).save(os.path.join(TMP_DIR, MOBILE_LAST_IMG_FILENAME))
        
        return format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path, type='punk_client')
    
@app.get("/get_image_by_path/")
async def process_image():
    file_path = os.path.join(TMP_DIR, MOBILE_LAST_IMG_FILENAME)

    print(f"return {file_path} file")

    return FileResponse(file_path)
        
if __name__ == '__main__':
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    uvicorn.run(app, host="0.0.0.0", port=PORT)