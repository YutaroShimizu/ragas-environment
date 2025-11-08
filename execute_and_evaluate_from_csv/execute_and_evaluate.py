import requests
import json
import time
import pickle
from ragas.testset.synthesizers.testset_schema import Testset
import os

import pandas as pd
from ragas import EvaluationDataset
from dotenv import load_dotenv

load_dotenv()

def send_query_to_rag_system(user_input):
    url = "https://oai-rag-dev-japaneast-010.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"

    payload = json.dumps({
    "messages": [
        {
        "role": "system",
        "content": "あなたは社内ナレッジに基づいて日本語で簡潔に回答するアシスタントです。根拠が無い場合はその旨を述べ、最後に参照元を示してください。"
        },
        {
        "role": "user",
        "content": f"{user_input}"
        }
    ],
    "temperature": 0.2,
    "max_tokens": 800,
    "data_sources": [
        {
        "type": "azure_search",
        "parameters": {
            "endpoint": "https://srch-rag-dev-japaneast-001.search.windows.net",
            "index_name": "azureblob-index",
            "authentication": {
            "type": "system_assigned_managed_identity"
            },
            "query_type": "semantic",
            "semantic_configuration": "default",
            "in_scope": True,
            "top_n_documents": 5,
            "strictness": 3,
            "fields_mapping": {
            "content_fields": [
                "content"
            ],
            "title_field": "title",
            "url_field": "url"
            },
            "include_contexts": [
            "citations"
            ]
        }
        }
    ]
    })
    headers = {
    'Content-Type': 'application/json',
    'api-key': os.environ["AZURE_OPENAI_API_KEY"]
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response

def execute_test():
    dataset = []
    df = pd.read_csv("user_input_reference.csv")

    for count, (_, row) in enumerate(df.iterrows(), start = 1):
        print(f"execute_test ({count}/{len(df)})")
        try:
            time.sleep(20)
            user_input = row["user_input"]
            reference = row["reference"]
            data = send_query_to_rag_system(user_input).json()
            response = data["choices"][0]["message"]["content"] # type: ignore
            retrieved_contexts = [doc["content"] for doc in data["choices"][0]["message"]["context"]["citations"]] # type: ignore
            dataset.append(
                 {
                    "user_input":user_input,
                    "retrieved_contexts":retrieved_contexts,
                    "response":response,
                    "reference":reference
                 }
            )
        except KeyError as e:
            print(f"KeyError{e}")
            print(data)
    eval_dataset = EvaluationDataset.from_list(dataset)
    EVAL_DATASET_FILEPATH = "eval_dataset"
    eval_dataset.to_pandas().map(
        lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x
    ).to_csv(
        f"{EVAL_DATASET_FILEPATH}.csv",index=False, encoding='utf-8-sig'
    )
    return eval_dataset

def evaluate(eval_dataset):
    from ragas.metrics import (
        context_precision,
        answer_relevancy,
        faithfulness,
        context_recall,
    )

    # list of metrics we're going to use
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    azure_configs = {
        "base_url": "https://oai-rag-dev-japaneast-010.openai.azure.com/",
        "model_deployment": "gpt-4o",
        "model_name": "gpt-4o",
        "embedding_deployment": "text-embedding-ada-002",
        "embedding_name": "text-embedding-ada-002",  # most likely
    }


    from langchain_openai.chat_models import AzureChatOpenAI
    from langchain_openai.embeddings import AzureOpenAIEmbeddings
    from ragas import evaluate

    azure_model = AzureChatOpenAI(
        openai_api_version="2025-01-01-preview", # type: ignore
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    )

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    azure_embeddings = AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15", # type: ignore
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    )

    result = evaluate(
        eval_dataset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings # type: ignore
    )

    df = result.to_pandas() # type: ignore

    df_cleaned = df.map(lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
    df_cleaned.to_csv('evaluation_result.csv', index=False, encoding='utf-8-sig')

def main():
    eval_dataset = execute_test()
    evaluate(eval_dataset)


main()