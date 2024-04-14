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

PORT = 8502

TMP_DIR = f'{workdir}tmp_files/'
WEIGHTS_DIR = f'{workdir}models/weights/'

app = FastAPI()

pipeline = Pipeline(WEIGHTS_DIR, TMP_DIR)
 
@app.get("/")
def root():
    return "Avialiable"


async def save_image(file_binary):
    img_file_name = f'tmp_{datetime.now()}_.jpg'
    img_file_path = os.path.join(TMP_DIR, img_file_name)

    print(img_file_path)

    async with aiofiles.open(img_file_path, 'wb') as out_file:
        await out_file.write(file_binary) 

    return img_file_path, file_binary
    

@app.post("/detect_punk_client/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
       
        img_file_path, binary_img_data = await save_image(file)

        classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        save_to_db(binary_img_data, img_file_path)

        response = format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path, type='punk_client')

        predict_img = Image.open(predict_img_path)
        bytes_image = io.BytesIO()

        predict_img.save(bytes_image, format='PNG')
        
        return Response(content=bytes_image.getvalue(), headers=response, media_type="image/png")

@app.post("/detect_punk_client_flattened_dataset/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
       
        img_file_path, binary_img_data = await save_image(file)

        pipeline = Pipeline(WEIGHTS_DIR, TMP_DIR, classifier_weights_name = 'weights_71_0.94_0.93_0.93_0.93.pt')

        classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        save_to_db(binary_img_data, img_file_path)

        response = format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path, type='punk_client')

        predict_img = Image.open(predict_img_path)
        bytes_image = io.BytesIO()

        predict_img.save(bytes_image, format='PNG')
        
        return Response(content=bytes_image.getvalue(), headers=response, media_type="image/png")

@app.post("/detect/")
async def process_image(file: bytes = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
    
        img_file_path, binary_img_data = await save_image(file)

        classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        save_to_db(binary_img_data, img_file_path)

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
    
        img_file_path, binary_img_data = await save_image(file)

        classifier_probs, recognited_text, predict_img_path = pipeline.forward(img_file_path)

        save_to_db(binary_img_data, img_file_path)
        
        return format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path, type='punk_client')
    
@app.post("/get_image_by_path/")
async def process_image(img_path: str):
    return FileResponse(img_path)
        
if __name__ == '__main__':
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    uvicorn.run(app, host="0.0.0.0", port=PORT)