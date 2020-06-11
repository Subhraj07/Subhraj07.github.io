---
layout: post
title:  "Updated model configure without shutting down services"
date:   2020-06-10 11:00:00
tags: Model Deployement
language: EN
---
Production deployement of Deep Learning models is very difficult. So we are going to see how we can create a very light weight asynchronous service using FastAPI and pytorch models (resnet and alexnet). The aim of this blog is to resolve the common problem in production where retrained model is configured in running service without shutting down.

Pretrained models : Neural Network models trained on large benchmark datasets like ImageNet.

<p style="text-align:center;"><img src="/images/model_deployement/Model_Timeline.png" style="width:75%"></p>

*Create fastapi service by using pretrained model(resnet and alexnet) in pytorch which will predict an image of a dog with probabilities of predicted classes.*

# FastAPI intro

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
* key features:

    * Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.
    * Fast to code: Increase the speed to develop features by about 200% to 300%. *
    * Fewer bugs: Reduce about 40% of human (developer) induced errors. *
    * Intuitive: Great editor support. Completion everywhere. Less time debugging.
    * Easy: Designed to be easy to use and learn. Less time reading docs.
    * Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
    * Robust: Get production-ready code. With automatic interactive documentation.
    * Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

Reference [FastAPI](https://fastapi.tiangolo.com/)

# Build Services

python dependecies need to install using pip: torch, torchvision, fastapi, uvicorn, Pillow
```
torch                       : Dynamic neural networks in Python
torchvision                 : consists of popular datasets, model architectures, and common image transformations for computer vision.
fastapi                     : fast web framework for building APIs
uvcorn                      : lightning-fast ASGI server
Pillow                      : Python Imaging Library 
```

*folder structure*
```
├── app.py
├── dog.jpg
├── imagenet_classes.txt
└── predict.py

app.py                      : fastapi service
predict.py                  : predict the probabilities of type of image and type of model
dog.jpg                     : Image of a dog
imagenet_classes.txt        : classes of pretrained model (resnet and alexnet)
```

code can be available [here](https://github.com/Subhraj07/blog_codes/tree/master/2020-06-10-production_model_reload_without_shutting_download_app)

# Code walkthrough

*predict.py*

import required packages

```
from torchvision import models, transforms
import torch
from PIL import Image
```

method -> predicted_results 

input parameter: image_path , pretrained model

read image using PIL
```
img = Image.open(img)
```

transform the input image so that they have the right shape and values should be similar to the ones which were used while training the model using torchvision module
```
transform = transforms.Compose([           
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])
```

pre-process the image
```
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
```

Load ImageNet classes
```
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]
```

get index of maximum output vector scores
```
out = pymodel(batch_t)
# Forth, print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
```

return model name and most probable classes
```
return {
        "model_name" : pymodel.__class__.__name__,
        "predicted_results" : [{classes[idx] : percentage[idx].item()} for idx in indices[0][:5]]
    }
```

Reference [here](https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/)

*app.py*

import requred packages and predict class dependencies

```
from fastapi import FastAPI
import uvicorn
from torchvision import models
from predict import predicted_results
```

intialise fastapi
```
app = FastAPI()
```

initialize resnet model for the app
```
pymodel = models.resnet101(pretrained=True)
```

call predict get method with img_path parameter and call predict method with the details
```
# query parameter http://127.0.0.1:8000/predict/?img_path=path/dog.jpg
@app.get("/predict/")
async def predict(img_path: str = ""):
    return predicted_results(img_path, pymodel)
```

update alexnet model without shutting down the application

call global model variable
update with new model
```
@app.post("/update_model")
async def read_item():
    global pymodel
    pymodel = models.alexnet(pretrained=True)
    return {"message": "model updated sucessfully"}
```

# Demo

#### steps:

* intialize service

<p style="text-align:center;"><img src="/images/model_deployement/1.png" style="width:75%"></p>

* predict image and see the result of resnet model

<p style="text-align:center;"><img src="/images/model_deployement/2.png" style="width:75%"></p>

* update the model to alexnet

<p style="text-align:center;"><img src="/images/model_deployement/3.png" style="width:75%"></p>

* predict image and see the result of alexnet model

<p style="text-align:center;"><img src="/images/model_deployement/4.png" style="width:75%"></p>