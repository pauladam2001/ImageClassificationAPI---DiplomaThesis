from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse
import uvicorn
from application.prediction import read_image, predict
from application.extract import face
import os


# The App is too large to be deployed on Heroku (because of tensorflow) (Heroku -> Resources -> Turn On web dyno + Deploy -> Enable Automatic Deploys)
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


@app.post('/api/extract')
async def extract_image(token, file: UploadFile = File(...)):
    if token == os.environ["TOKEN"]:
        extensions = file.filename.split(".")[1] in ("jpg", "jpeg", "png")
        if not extensions:
            return "Image doesn't have the right format! (.jpg, .jpeg, .png)"

        for file_name in os.listdir(os.getcwd()):
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                os.remove(file_name)                                      # delete the current images

        file_location = f"{os.getcwd()}/{file.filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(file.file.read())                           # save the image so we can read it with opencv

        response = face(f"{os.getcwd()}/{file.filename}")                 # extract the face and save the image

        return response                                                   # return the extracted image
    else:
        return "Token not valid!"


uvicorn.run(app)
