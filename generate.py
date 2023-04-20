import pprint

import torch
from torch import optim
from torchvision.utils import make_grid
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load the saved model
model = torch.load("models/my_diffusion_model.ckpt")

# Load the pre-trained model and tokenizer for encoding the prompt and negative prompt
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_pt = AutoModel.from_pretrained("gpt2")

def encode_prompt(prompt):
    prompt = " ".join(prompt.split())
    prompt_encoded = tokenizer.encode(prompt, return_tensors='pt')
    prompt_embedding = model_pt(prompt_encoded)[0][:, 0, :]
    return prompt_embedding

def generate_image(prompt, negative_prompt, steps, num_samples, scale, seed, strength):
    with torch.no_grad():
        text = tokenizer.encode(prompt)
        text_neg = tokenizer.encode(negative_prompt)
        noise_prompt = torch.randn([1, 1, 256, 256]).cuda()
        noise_strength = strength
        optimizer = optim.Adam(noise_prompt.requires_grad_(True), lr=0.1)
        steps = steps

        for step in range(steps):
            optimizer.zero_grad()
            out = model(noise_prompt, text, text_neg, scale)
            loss = noise_strength * out.mean()
            loss.backward()
            optimizer.step()

        z = torch.randn((num_samples, 1, 256, 256), device="cuda")
        images = model.decoder(z, text, temperature=0.7)
        return images