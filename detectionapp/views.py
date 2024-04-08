from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from PIL import Image
from io import BytesIO
from .utils import predict, convert_to_corners  # Importing the function from utils.py
import traceback
import torch

def index(request):
    """Render the main page with the webcam feed and detection button."""
    return render(request, 'detectionapp/index.html')

@csrf_exempt
def detect(request):
    try:
        # Parse the incoming JSON data
        data = json.loads(request.body)
        
        # Extract the base64 encoded image data (remove the "data:image/jpeg;base64," prefix)
        image_data = data['image'].split(',')[1]
        
        # Decode the base64 data to get the image bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert the image bytes to a PIL Image
        image = Image.open(BytesIO(image_bytes))
        print(f"Image loaded with size: {image.size}") # Log 1

        # TODO: Additional preprocessing if required

        # Run the image through the YOLOv5 model
        img_with_boxes, gender_preds = predict(image)
        
        # Convert numpy array to PIL Image
        img_with_boxes_pil = Image.fromarray(img_with_boxes)
        
        # Convert PIL Image to bytes
        buffered = BytesIO()
        img_with_boxes_pil.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        image_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Log success and gender predictions
        print("Detection successful.")
        print("Gender Predictions:", gender_preds)

        return JsonResponse({"image": image_base64, "gender_predictions": gender_preds})
        
    except Exception as e:
        error_message = f"Error during detection: {e}"
        traceback_message = traceback.format_exc()
        print(traceback_message)
        
        return JsonResponse({"error": traceback_message}, status=500)
