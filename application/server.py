from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import uvicorn
from application.prediction import read_image, predict
import os

app = FastAPI(title='Image Classification API')


@app.get('/')
async def index():
    return RedirectResponse(url="docs")


@app.post('/api/predict')
async def predict_image(token, file: UploadFile = File(...)):
    if token == os.environ["TOKEN"]:
        extensions = file.filename.split(".")[1] in ("jpg", "jpeg", "png")          # check if the image has one of the three extensions
        if not extensions:
            return "Image doesn't have the right format! (.jpg, .jpeg, .png)"

        image = read_image(await file.read())                                       # read the image

        return predict(image)                                                       # return the predictions
    else:
        return "Token not valid!"


uvicorn.run(app)
