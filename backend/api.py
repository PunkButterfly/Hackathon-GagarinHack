from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import os

import aiofiles
from datetime import datetime

import clickhouse_connect

import yaml

from utils import *

from models.classifier import TorchClassifier

# workdir = './backend/'
workdir = '' # for docker 

# with open(f'{workdir}CH_CONFIG.yaml', 'r')  as f:
#     config = yaml.safe_load(f)

# CH_HOST = config['HOST']
# CH_PORT = config['PORT']
# CH_USERNAME = config['USER']
# CH_PASSWORD = config['PASSWORD']

TMP_DIR = f'{workdir}tmp_files/'

if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

PORT = 8502

app = FastAPI()

# def save_to_db(binary_data, file):
#     client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, username=CH_USERNAME, password=CH_PASSWORD)

#     s_bin = str(binary_data)[2:-1]

#     client.command("INSERT INTO GagarinHack2024.queries (bindata, filename) VALUES ('{}', '{}')".format(s_bin, file.filename))
    
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

        model = TorchClassifier(f'{workdir}models/weights/v3_weights.pt')

        img_result = model.process_img(out_file_path)

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
            await out_file.write(binary_data)  #

        # save_to_db(binary_data, file)

        model = TorchClassifier(f'{workdir}models/weights/v3_weights.pt')

        img_result = model.process_img(out_file_path)

        parsed_result = parse_res(img_result)

        formatted_result = {
            "type": parsed_result[1][0],
            "confidence": parsed_result[0],
            "series": '1488',
            "number": '228 228',    
            "page_number": parsed_result[1][1]
        }

        return formatted_result



# @app.post("/process_image/")
# async def process_image(file: UploadFile = File(...)):

#     # return {"filename": file.filename}
#     if not file_data:
#         return {"message": "No upload file sent"}
#     elif not allowed_img(file_name):
#         return {"message": "Not allowed file extension"}
#     else:
#         out_file_name = f'tmp_{datetime.now()}_' + file_name
#         out_file_path = os.path.join(TMP_DIR, out_file_name)

#         async with  aiofiles.open(out_file_path, 'wb') as out_file:
#             binary_data = file_data 
#             await out_file.write(binary_data)  #

#         # save_to_db(binary_data, file)

#         model = TorchClassifier(f'{workdir}models/weights/v2_weights.pt')

#         img_result = model.process_img(out_file_path)
#         print(img_result)

#         return img_result
        
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)