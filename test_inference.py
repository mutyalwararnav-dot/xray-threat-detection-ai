import requests
import numpy as np
from PIL import Image
import io

print("Generating test image...")
np.random.seed(42)
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)

buf = io.BytesIO()
img.save(buf, format='JPEG')
byte_img = buf.getvalue()

print("Sending POST request to Firebase local emulator (http://localhost:8080)...")
try:
    response = requests.post("http://localhost:8080", data=byte_img, headers={'Content-Type': 'application/octet-stream'})
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except:
        print("Response Body:", response.text)
except Exception as e:
    print("Error:", e)
