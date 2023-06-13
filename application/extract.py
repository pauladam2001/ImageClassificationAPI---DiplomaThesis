import cv2      # an image processing module
import dlib     # C++ toolkit containing machine learning algorithms
import os
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_NAME"],
    api_key=os.environ["CLOUDINARY_KEY"],               # cloudinary credentials
    api_secret=os.environ["CLOUDINARY_SECRET"]
)

detector = dlib.get_frontal_face_detector()     # dlib's frontal face detector
path = f"{os.getcwd()}/face"


def save(image, name, box, width=150, height=150):
    x, y, w, h = box
    image_crop = image[y:h, x: w]
    image_crop = cv2.resize(image_crop, (width, height))
    cv2.imwrite(name + ".jpg", image_crop)                                                          # save the new image with the face


def face_extraction(file):
    frame = cv2.imread(file)
    faces = detector(frame)
    margin = 50

    if len(faces) != 1:                                                                             # only one face allowed
        return "Please upload a valid ID Card!"
    else:
        for face in faces:
            x1, y1 = face.left(), face.top()                                                        # get face coordinates
            x2, y2 = face.right(), face.bottom()
            save(frame, path + str(0), (x1 - margin, y1 - margin, x2 + margin, y2 + margin))        # save a new picture with the face (with padding)

        frame = cv2.resize(frame, (800, 800))

        # return FileResponse("face0.jpg")      # before we returned the file, now we return the link to cloudinary

        result = cloudinary.uploader.upload("face0.jpg")
        url = result.get("url")
        return url                      # upload image directly to cloudinary
