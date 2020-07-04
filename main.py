import json
import os
import time
from urllib.request import urlopen

from dotenv import load_dotenv
from transformers import T5Tokenizer, TFT5ForConditionalGeneration


def deg2dir(deg):
  return ["north","north-east","east","south-east","south","south-west","west","north-west"][int((deg/45)+.5) % 8]

model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

load_dotenv()

with urlopen(f"http://api.ipstack.com/check?access_key={os.getenv('LOCATION_API')}") as location_file:
  location_data = json.load(location_file)
  lat, lon = location_data['latitude'], location_data['longitude']

with urlopen(f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={os.getenv('WEATHER_API')}") as file:
  data = json.load(file)

print(data)

weather = 'Weather can be described as'
for event in data['weather']:
  weather += " " + event['description'] + ','
  
text = f"""
Here, in {data['name']} the temperature is {round(data['main']['temp'])} degrees Celcius but it feels like {round(data['main']['feels_like'])} degrees Celcius.
Pressure is {data['main']['pressure']} hPa and humidity is at {data['main']['humidity']}%.
Wind flows from {deg2dir(data['wind']['deg'])} at {data['wind']['speed']} m/s.
Sun rises at {time.strftime('%H:%M', time.gmtime(data['sys']['sunrise']))} and sets at {time.strftime('%H:%M', time.gmtime(data['sys']['sunset']))}.
{weather}
"""

print(text)

preprocess_text = text.strip().replace("\n"," ")
tokenized_text = tokenizer.encode("summarize: " + preprocess_text, return_tensors="tf")

summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=20, max_length=80, early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"Summarized text:\n{output}")
