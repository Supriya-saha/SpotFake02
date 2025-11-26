import random
from google import genai
from google.genai import types

# client = genai.Client(api_key="AIzaSyCcVyjZReOnPqPLFcVDDwN70j95MWeu8KI")  # replace after regenerating

# # Load image bytes
# with open("gradcam_sample_0.png", "rb") as f:
#     img_bytes = f.read()

# image_part = types.Part.from_bytes(
#     data=img_bytes,
#     mime_type="image/png"  # change to image/jpeg if JPG
# )

# # Send image + text
# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents=[
#         "Describe what you see in this image.",
#         image_part
#     ],
# )
x=(random.uniform(0, 20))
x=round(x, 2)
print(x)
print(round(random.uniform(0,20),2))
# print(response.text)
