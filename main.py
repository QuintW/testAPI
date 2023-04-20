import pprint

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, Request, Form, File
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

from generate import generate_image

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class teststr(BaseModel):
    test: str

class GeneratedImgJSON(BaseModel):
    prompt: str
    negative_prompt: str = None
    steps: int
    num_samples: int
    scale: int
    seed: int
    strength: int

class GeneratedDepthJSON(BaseModel):
    input_image: UploadFile

class GenerateImgToImgJSON(BaseModel):
    input_image: UploadFile = None
    prompt: str
    negative_prompt: str = None
    steps: int
    num_samples: int
    scale: int
    seed: int
    strength: int
    depth_image: UploadFile = None


@app.post("/users")
async def get_user(user: int = Form(...)):
    # create user in database or perform other logic
    return {"message": "User created", "user": "test"}


class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str
    steps: int
    num_samples: int
    scale: int
    seed: int
    strength: int


@app.post("/get-promptA")
async def create_img(request: Request):
    form_data = await request.form()
    prompt = form_data.get('prompt')
    negative_prompt = form_data.get('negative_prompt')
    steps = form_data.get('steps')
    num_samples = form_data.get('num_samples')
    scale = form_data.get('scale')
    seed = form_data.get('seed')
    strength = form_data.get('strength')

    # Generate the image
    image = generate_image(prompt, negative_prompt, steps, num_samples, scale, seed, strength)

    return {
        "message": "Image created",
        "image": image.tolist(),
    }

@app.post("/get-promptB")
async def get_promptB(request: Request):
    form_data = await request.form()
    prompt = form_data.get('prompt')
    negative_prompt = form_data.get('negative_prompt')
    steps = form_data.get('steps')
    num_samples = form_data.get('num_samples')
    scale = form_data.get('scale')
    seed = form_data.get('seed')
    strength = form_data.get('strength')
    input_image = form_data.get('input_image')
    depth_image = form_data.get('depth_image')

    return {
        "message": "Prompt created",
        "postWay": "get_promptB",
        "form_data": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "num_samples": num_samples,
            "scale": scale,
            "seed": seed,
            "strength": strength,
            "input_image": input_image,
            "depth_image": depth_image
        }
    }

@app.post("/get-promptC")
async def get_promptC(request: Request):
    pprint.pprint(vars(request))
    form_data = await request.form()
    pprint.pprint(form_data)
    prompt = form_data.get('prompt')
    negative_prompt = form_data.get('negative_prompt')
    steps = form_data.get('steps')
    num_samples = form_data.get('num_samples')
    scale = form_data.get('scale')
    seed = form_data.get('seed')
    strength = form_data.get('strength')
    input_image = form_data.get('input_image')
    depth_image = form_data.get('depth_image')

    return {
        "message": "Prompt created",
        "postWay": "get_promptC",
        "form_data": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "num_samples": num_samples,
            "scale": scale,
            "seed": seed,
            "strength": strength,
            "input_image": input_image,
            "depth_image": depth_image
        }
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("main.html", {"request": request, "images":"geen"})



# @app.get("/generate_img/{test}")
# async def generate_img(test: teststr):
#     pprint.pprint(test)
#     url = 'test1'
#     return {"url": url}

# @app.get("/generate_depth/{data}")
# async def generate_depth(data: GeneratedDepthJSON):
#     pprint.pprint(data)
#     url = 'test2'
#     return {"url": url}

# @app.get("/generate_img_to_img/{data}")
# async def generate_img_to_img(data: GenerateImgToImgJSON):
#     pprint.pprint(data)
#     url = 'test3'
#     return {"url": url}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000) # uvicorn main:app --reload