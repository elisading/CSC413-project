import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

bike_prompts = [
    "Bike toy advertisement", "Kids riding bike toy advertisement", "a <bike-toy> advertisement in a toy store, realistic", 
    "kids riding <bike-toy> advertisement, realistic, toy bicycle with two wheels, playful children 5-7 years old"]
lora_bike_pics = ["lora_bike_prompt1.png","lora_bike_prompt2.png","lora_bike_prompt4.png","lora_bike_prompt4.png"]
ti_bike_pics = ["ti_bike_prompt1.png", "ti_bike_prompt2.png"]
db_bike_pics = ["C-DB.png", "D-DB.png"]
db_dino_pics = ["A-DB.png", "B-DB.png"]

dino_prompts = [
    "Dinosaur toys in a store advertisement", "Kids playing with dinosaur toys advertisement",
    "a collection of <dino-toys>, realistic, advertisement, happy"
    "curious children looking at  <dino-toys> in a toy store, realistic, advertisement, happy, laughing, playful, intrigued, shopping"
    "<dino-toys> in a toy store, realistic, advertisement, happy, shopping, packaging close up"]
lora_dino_pics = ["lora_dino_prompt1.png","lora_dino_prompt2.png","lora_dino_prompt3.png","lora_dino_prompt4.png","lora_dino_prompt5.png"]
ti_dino_pics = ["ti_dino_prompt1.png", "ti_dino_prompt2.png"]


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
avg_sim = 0

prompts = dino_prompts
pics = lora_dino_pics

for prompt, pic in zip(prompts, pics):
    image = preprocess(Image.open(pic)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        cos_sim = cosine_similarity(image_features, text_features)
        #  print("cosine similarity:", cosine_similarity)
        avg_sim += cos_sim
avg_sim /= len(pics)
print("avg cosine similarity", avg_sim)
