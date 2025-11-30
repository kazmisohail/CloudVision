from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .model_loader import ModelLoader
from .utils import preprocess_v1, preprocess_v2, preprocess_v3, remap_classes, mask_to_base64, mitigate_shadows, image_to_base64
import numpy as np
import google.generativeai as genai
import os
import json

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.data.get('file')
        model_type = request.data.get('model_type', 'v2') # Default to v2

        if not file_obj:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Load Model
            loader = ModelLoader()
            model = loader.load_model(model_type)
            
            if not model:
                 return Response({'error': f'Model {model_type} not loaded'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            # Preprocess based on model type
            if model_type == 'v1':
                input_tensor = preprocess_v1(file_obj)
            elif model_type == 'v3':
                input_tensor = preprocess_v3(file_obj)
            else:
                input_tensor = preprocess_v2(file_obj)

            # Predict
            prediction = model.predict(input_tensor)
            
            # Remap Classes
            remapped_mask = remap_classes(prediction)
            
            # Calculate Percentages
            total_pixels = remapped_mask.size
            counts = np.bincount(remapped_mask.flatten(), minlength=4)
            percentages = {
                "Clear": (counts[0] / total_pixels) * 100,
                "Shadow": (counts[1] / total_pixels) * 100,
                "Thin Cloud": (counts[2] / total_pixels) * 100,
                "Thick Cloud": (counts[3] / total_pixels) * 100
            }
            
            has_shadow = percentages["Shadow"] > 1.0
            
            # Convert Mask to Base64 (Grayscale)
            mask_b64 = mask_to_base64(remapped_mask)
            
            return Response({
                'mask': mask_b64,
                'percentages': percentages,
                'has_shadow': has_shadow,
                'model_used': model_type
            })

        except Exception as e:
            print(f"Prediction Error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class MitigateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.data.get('file')
        # Ideally we should pass model_type here too, but for now we'll default to v2 
        # or use the one that's likely cached/loaded.
        # To be safe, let's default to v2 as it's the advanced one.
        model_type = request.data.get('model_type', 'v2') 
        
        if not file_obj:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            # Re-predict to get the mask
            loader = ModelLoader()
            model = loader.load_model(model_type)
            if not model:
                 return Response({'error': 'Model not loaded'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

            if model_type == 'v1':
                input_tensor = preprocess_v1(file_obj)
            else:
                input_tensor = preprocess_v2(file_obj)
                
            prediction = model.predict(input_tensor)
            remapped_mask = remap_classes(prediction)
            
            # Mitigate
            mitigated_img = mitigate_shadows(file_obj, remapped_mask)
            mitigated_b64 = image_to_base64(mitigated_img)
            
            return Response({'mitigated_image': mitigated_b64})
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GeminiAnalysisView(APIView):
    def get(self, request, *args, **kwargs):
        return Response({'status': 'Gemini Analysis Endpoint Ready. Send POST request with class percentages.'})

    def post(self, request, *args, **kwargs):
        percentages = request.data.get('percentages')
        if not percentages:
            return Response({'error': 'No percentages provided'}, status=status.HTTP_400_BAD_REQUEST)
            
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
             return Response({'error': 'Gemini API Key not configured'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
             
        try:
            genai.configure(api_key=api_key)
            # Use gemini-2.5-flash-lite
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            prompt = (
                f"Analyze these satellite cloud metrics: {json.dumps(percentages)}. "
                "Provide a meteorological report in the following format:\n"
                "1. Small Paragraph: General overview.\n"
                "2. AI Suggestions: Actionable advice.\n"
                "3. Future Prediction: Short-term forecast.\n"
                "4. Precautions: Safety measures.\n"
                "Do not use special characters like asterisks or markdown bolding. Keep it clean text."
            )
            
            response = model.generate_content(prompt)
            return Response({'analysis': response.text})
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Return the actual error to the frontend for debugging
            return Response({'analysis': f"AI Analysis Error: {str(e)}"}, status=status.HTTP_200_OK)
