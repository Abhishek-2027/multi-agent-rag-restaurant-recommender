import requests
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

synth_db_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/nNGhXQQi2CIw9JcInMe67Q/Synthetic-Restaurants-Cafes-Bakeries.txt"

response = requests.get(synth_db_url)
restaurant_db = response.json()

documents = []
doc_id = 0

for source in restaurant_db.keys():
    for shop in restaurant_db[source]:

        signature = "\n".join(
            f"Dish {i+1}: {dish}"
            for i, dish in enumerate(shop["signature_items"])
        )

        reviews = "\n".join(
            f"Review {i+1}. Title: {shop['review_titles'][i]}. Text: {shop['reviews'][i]}"
            for i in range(len(shop["reviews"]))
        )

        content = (
            f"Shop name: {shop['label']}. "
            f"Shop type: {shop['type']}. "
            f"Shop location: {shop['location']}. "
            f"Shop rating: {shop['rating']}. "
            f"Shop price_range: {len(shop['price_range']) // 3}. "
            f"Description: {shop['short_description']}.\n\n"
            f"Signature dishes:\n{signature}\n\n"
            f"Reviews:\n{reviews}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={"source": source, "id": doc_id}
            )
        )
        doc_id += 1

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = Chroma(
    collection_name="RestaurantDB",
    embedding_function=embedding_model
)

vector_db.add_documents(documents)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("✅ Chroma DB and retriever are ready")
