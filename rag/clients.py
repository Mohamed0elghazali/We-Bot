### HANDLE ENVIRONMENT
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r".env")

from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

llm = ChatBedrockConverse(
    model=os.getenv("AWS_Nova_Pro"), # AWS_LLAMA_3_3 - AWS_Nova_Pro
    temperature=0,
    max_tokens=2048,
    top_p=0.4,
    region_name=os.getenv("AWS_REGION")
)

embeddings = BedrockEmbeddings(
    model_id=os.getenv("AWS_TITAN_EMBED_V2"),
    region_name=os.getenv("AWS_REGION")
)