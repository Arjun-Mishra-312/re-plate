import os
import cv2
from datetime import datetime, timedelta
from dateutil.parser import parse
from pydantic import BaseModel
import json
import re
from typing import Optional
import google.generativeai as genai
from PIL import Image

# Initialize Google Gemini
GOOGLE_API_KEY = "AIzaSyD91wZF6ytN19uT-EW3tMPr3oOmY4yNZAo"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Enhanced expiry date parser
def parse_date(texts):
    combined = " ".join(texts)
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',   # YYYY/MM/DD
        r'\b\d{6}\b'                          # DDMMYY format
    ]
    for pattern in patterns:
        matches = re.findall(pattern, combined)
        for m in matches:
            try:
                if len(m) == 6 and m.isdigit():
                    return datetime.strptime(m, "%d%m%y")
                return parse(m, dayfirst=True, fuzzy=True)
            except Exception:
                continue
    return None

def categorize(expiry_date):
    today = datetime.today().date()
    if expiry_date < today:
        return "Expired"
    elif expiry_date <= today + timedelta(days=5):
        return "Critical"
    else:
        return "Good"

class ExpiryInfo(BaseModel):
    expiry_date: Optional[str]

def get_expiry_info(text: str) -> ExpiryInfo:
    expiry_date = parse_date([text])
    if expiry_date:
        return ExpiryInfo(expiry_date=expiry_date.strftime("%Y-%m-%d"))
    else:
        return ExpiryInfo(expiry_date=None)

def scan_image(image_path):
    try:
        image = Image.open(image_path)
        prompt = "Extract the expiry date from this image. If there is no date, return null."
        response = model.generate_content([prompt, image])
        response.resolve()
        extracted_text = response.text
        print(f"[Gemini Extracted Text] {extracted_text}")

        expiry_info = get_expiry_info(extracted_text)

        if expiry_info and expiry_info.expiry_date:
            expiry_date = parse(expiry_info.expiry_date).date()
            status = categorize(expiry_date)
            print(f"[Expiry Date]: {expiry_date} | [Status]: {status}")
            return {"expiry_date": str(expiry_date), "status": status}
        else:
            print("⚠️ No valid expiry date found.")
            return None
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return None

def main():
    file_path = "Sample_Images/test.jpg"
    
    if not os.path.exists(file_path):
        print(f"⚠️ File '{file_path}' not found in the current directory.")
        return
    
    print(f"📂 Processing file: {file_path}")
    
    result = scan_image(file_path)
    
    if result:
        print(result)

if __name__ == "__main__":
    main()
