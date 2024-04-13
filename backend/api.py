from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
import uvicorn
import os

import aiofiles
from datetime import datetime

import clickhouse_connect

import yaml

from utils import *

from models.classifier import TorchClassifier
from models.Pipeline import Pipeline

# workdir = './backend/'
workdir = '' # for docker 

PORT = 8502

TMP_DIR = f'{workdir}tmp_files/'
WEIGHTS_DIR = f'{workdir}models/weights/'

app = FastAPI()

classifier_model = TorchClassifier(f'{WEIGHTS_DIR}v3_weights.pt')
pipeline = Pipeline(WEIGHTS_DIR, TMP_DIR)
 
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
            await out_file.write(binary_data) 

        # save_to_db(binary_data, file)

        img_result = classifier_model.process_img(out_file_path)

        return img_result
    

@app.post("/detect/")
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
            await out_file.write(binary_data)  

        # save_to_db(binary_data, file)

        img_result = classifier_model.process_img(out_file_path)

        parsed_result = parse_res(img_result)

        text, img_path = pipeline.forward(out_file_path)
        print(text, img_path)

        formatted_result = {
            "type": parsed_result[1][0],
            "confidence": parsed_result[0],
            "series": '1331',
            "number": '111 111',    
            "page_number": parsed_result[1][1],
            "proceed_image_name": img_path
        }

        return formatted_result
    
@app.post("/get_image_by_path/")
async def process_image(img_path: str):
    return FileResponse(img_path)
        
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)

    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)