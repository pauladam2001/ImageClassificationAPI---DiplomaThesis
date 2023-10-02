These 2 APIs are used for **BikeFiesta**, which is my diploma thesis, a marketplace for used bicycles.

The **Image Classification API** is responsible for receiving an image and returning a list of predictions of what objects the image contains. It uses a pre-trained **ResNet101V2** neural network trained on the **ImageNet** dataset, a dataset of over 1 million images and 1000 classes. The API receives an image in the form of a URL to a **Cloudinary** hosted image, which is then downloaded and passed through a pre-trained neural network to make predictions on the image.

The **Face Extraction API** allows users to extract a profile picture from an ID Card. It is designed to be user-friendly, efficient, and reliable. With its simple input and output formats, it is easy to integrate into my used bicycles marketplace. The API uses **Dlib** to perform face detection. The user simply uploads an image of an ID card and submits a valid token. The API then extracts the profile picture from the ID card, crops and resizes it, and returns a Cloudinary URL to the extracted image.

To learn more about these APIs and how I used them in my Ruby on Rails application, please see the following:
- [BikeFiesta - GitHub](https://github.com/pauladam2001/BikeFiesta-DiplomaThesis)
- [BikeFiesta - A marketplace for used bicycles.pdf](https://github.com/pauladam2001/BikeFiesta-DiplomaThesis/files/12785265/Licenta.IE.ADAM.VA.PAUL-ADRIAN.pdf)
- [BikeFiesta - A marketplace for used bicycles - Presentation.pdf](https://github.com/pauladam2001/BikeFiesta-DiplomaThesis/files/12785269/BikeFiesta.-.A.marketplace.for.used.bicycles.-.Slides.pdf)
