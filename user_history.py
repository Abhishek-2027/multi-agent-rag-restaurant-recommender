# user_history.py
# -----------------------------------------------------------
# This file:
# - Loads real user reviews
# - Supports multiple users (via userId)
# - Cleans reviews
# - ANALYZES IMAGES (OpenAI Vision)
# - Returns enriched visit_history for ONE user
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import requests
import ast
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------
# LOAD DATASETS
# -----------------------------------------------------------

REVIEWS_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/8R-csw5FyfI1nomsQeniNA/review-user.csv'
ITEM_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rZLLfu0XtsvS4Jn1mSfngQ/restaurant-item.csv'

target_user_reviews = pd.read_csv(REVIEWS_URL)
restaurant_info = pd.read_csv(ITEM_URL)

# -----------------------------------------------------------
# PREPROCESSING HELPERS
# -----------------------------------------------------------

price_interval_map = {np.nan: 0, '$': 1, '$$ - $$$': 2, '$$$$': 3}
restaurant_info['priceInterval'] = restaurant_info['priceInterval'].map(price_interval_map)

def rating_map(rating):
    """Normalize rating from 0–10 to 0–1."""
    return rating / 10.0

def itemId_map(itemId):
    """Convert itemId into readable restaurant description."""
    rest = restaurant_info[restaurant_info['itemId'] == itemId]
    if len(rest) == 0:
        return "Restaurant info unavailable."

    return (
        f"This is a {rest['type'].iloc[0]} restaurant with "
        f"price tier {rest['priceInterval'].iloc[0]} "
        f"and average rating {rest['rating'].iloc[0] / 10.0}."
    )

# -----------------------------------------------------------
# IMAGE HANDLING
# -----------------------------------------------------------

# Vision-capable OpenAI model
vision_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def parse_images(images_field):
    """
    Convert string representation of image list into Python list.
    Example: "['url1','url2']" -> ['url1','url2']
    """
    if not images_field or images_field == '[]':
        return []
    try:
        return ast.literal_eval(images_field)
    except Exception:
        return []

from openai import OpenAI

client = OpenAI()

def analyze_images(image_urls):
    descriptions = []

    if not image_urls:
        return ["No images provided."]

    for idx, url in enumerate(image_urls):
        print(f"Analyzing image {idx+1}/{len(image_urls)}")
        try:
            img = requests.get(url, timeout=10).content
            encoded = base64.b64encode(img).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the food or dining environment in one short sentence."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50
            )

            descriptions.append(response.choices[0].message.content)
            print("✓ Image analyzed")

        except Exception as e:
            print("✗ Image failed:", e)
            descriptions.append("Image could not be analyzed.")

    return descriptions

# -----------------------------------------------------------
# MAIN FUNCTION: USER-SPECIFIC VISIT HISTORY
# -----------------------------------------------------------

def get_user_history(user_id: str):
    """
    Return full visit history for ONE specific user,
    enriched with image descriptions.
    """
    df = target_user_reviews[target_user_reviews["userId"] == user_id]

    if df.empty:
        return []

    # Clean & normalize fields
    df['rating'] = df['rating'].apply(rating_map)
    df['itemId'] = df['itemId'].apply(itemId_map)

    df = df.rename(columns={
        'itemId': 'restaurant info',
        'title': 'review title',
        'text': 'review text'
    })

    records = df.to_dict(orient="records")

    # 🔥 IMAGE ANALYSIS PER VISIT (USER-SPECIFIC)
    for record in records:
        image_urls = parse_images(record.get("images", "[]"))
        record["image_descriptions"] = analyze_images(image_urls)

    print("completed user history + image analysis")
    return records

# -----------------------------------------------------------
# TEST LOCALLY
# -----------------------------------------------------------

if __name__ == "__main__":
    history = get_user_history('DAE0922BB12703E868929052C6CD433A')
    print("Returned records:", len(history))
    print(history[:1])  # print only first record
