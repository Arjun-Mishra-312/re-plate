#!/usr/bin/env python

import sys
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
# It will now first check the loaded .env variables, then system environment
try:
    api_key = os.getenv("GOOGLE_API_KEY") # Use os.getenv to avoid KeyError if not found
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=api_key)
except ValueError as e:
    print(f"Error: {e}")
    print("Please ensure GOOGLE_API_KEY is set in your .env file or environment variables.")
    sys.exit(1)
except Exception as e: # Catch other potential configuration errors
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

def main(image_path):
    # Try opening the image file.
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash') # Use the latest flash model

    # Define classification labels and the prompt for Gemini
    condition_labels = ["safe for consumption", "needs immediate distribution", "waste"]
    food_type_labels = ["Fresh Produce", "Dairy & Eggs", "Proteins", "Grains & Pantry Staples", "Canned & Packaged Goods"]
    
    # Updated prompt to ask for food type categorization as well
    prompt = f"""
Analyze the food item in the image. Classify it into three categories:

1. FOOD NAME - Identify the specific name of the food item (e.g., banana, cheddar cheese, chicken breast).

2. FOOD CONDITION - Choose one of the following:
   - **safe for consumption**: The food looks fresh and suitable for eating.
   - **needs immediate distribution**: The food is slightly aged, bruised, or nearing spoilage but still edible. It should be distributed quickly.
   - **waste**: The food shows clear signs of spoilage like mold, significant rot, or decay and is not suitable for consumption.

3. FOOD TYPE - Choose one of the following based on the identified food name:
   - **Fresh Produce**: Fruits, vegetables, fresh herbs
   - **Dairy & Eggs**: Milk, cheese, yogurt, eggs, butter
   - **Proteins**: Meat, poultry, fish, tofu, beans
   - **Grains & Pantry Staples**: Bread, rice, pasta, flour, sugar
   - **Canned & Packaged Goods**: Canned foods, boxed items, packaged snacks

Respond with all three classifications and a brief explanation for your choice for Condition but not Name or Type.
Use the following format exactly:
Name: [Specific food name]
Condition: [one of '{condition_labels[0]}', '{condition_labels[1]}', or '{condition_labels[2]}']
Type: [one of '{food_type_labels[0]}', '{food_type_labels[1]}', '{food_type_labels[2]}', '{food_type_labels[3]}', or '{food_type_labels[4]}']
Reason: [Your brief explanation]
"""

    # Generate content using the image and prompt
    try:
        response = model.generate_content([prompt, img], stream=False)
        response.resolve() # Wait for the response to complete
        response_text = response.text.strip()

        # Parse the response to extract condition, type and reason
        condition = "Unknown"
        food_type = "Unknown"
        reason = "No reason provided."
        name = "Unknown" # Initialize name variable

        try:
            lines = response_text.split('\n')
            for line in lines:
                if line.lower().startswith("name:"): # Extract Name
                    name = line.split(":", 1)[1].strip()
                elif line.lower().startswith("condition:"):
                    condition = line.split(":", 1)[1].strip()
                elif line.lower().startswith("type:"):
                    food_type = line.split(":", 1)[1].strip()
                elif line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
        except Exception as parse_error:
            print(f"Warning: Could not parse model response: {parse_error}")
            print(f"Raw model response:\n{response_text}")
            # Attempt basic extraction if parsing fails
            # (Simple name extraction might be difficult/unreliable here, skipping for now)
            for l in condition_labels:
                if l in response_text.lower():
                    condition = l
                    break
            for t in food_type_labels:
                if t.lower() in response_text.lower():
                    food_type = t
                    break

        # Validate the extracted condition
        if condition not in condition_labels and condition != "Unknown":
            print(f"Warning: Model returned an unexpected condition: '{condition}'")
            # Simple fallback: check if any known label is in the raw text
            found_condition = None
            for l in condition_labels:
                if l in response_text.lower():
                    found_condition = l
                    break
            if found_condition:
                condition = found_condition
                print(f"Attempted to extract condition: {condition}")
        
        # Validate the extracted food type
        if food_type not in food_type_labels and food_type != "Unknown":
            print(f"Warning: Model returned an unexpected food type: '{food_type}'")
            # Simple fallback: check if any known type is in the raw text
            found_type = None
            for t in food_type_labels:
                if t.lower() in response_text.lower():
                    found_type = t
                    break
            if found_type:
                food_type = found_type
                print(f"Attempted to extract food type: {food_type}")

        # Print the result with condition, type and reason
        print(f"Name: {name}") # Print the name
        print(f"Condition: {condition}")
        print(f"Food Type: {food_type}")
        print(f"Reason: {reason}")

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure dotenv is installed
    try:
        import dotenv
    except ImportError:
        print("Error: 'python-dotenv' library not found.")
        print("Please install it using: pip install python-dotenv")
        sys.exit(1)

    if len(sys.argv) < 2:
        # Updated usage message
        print("Usage: python foodClassifier.py <path_to_image>")
        sys.exit(1)

    # Ensure the google-generativeai library is installed
    try:
        import google.generativeai
    except ImportError:
        print("Error: 'google-generativeai' library not found.")
        print("Please install it using: pip install google-generativeai")
        sys.exit(1)

    main(sys.argv[1]) 