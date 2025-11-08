from langchain_community.document_loaders import DirectoryLoader
import asyncio
from dotenv import load_dotenv

path = "source"
loader = DirectoryLoader(path, glob="**/*.pdf")
docs = loader.load()

load_dotenv()

# other configuration
azure_configs = {
    "base_url": "https://oai-rag-dev-japaneast-010.openai.azure.com/",
    "model_deployment": "gpt-4o",
    "model_name": "gpt-4o",
    "embedding_deployment": "text-embedding-ada-002",
    "embedding_name": "text-embedding-ada-002",  # most likely
}

async def main():
    from langchain_openai import AzureChatOpenAI
    from langchain_openai import AzureOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    generator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        openai_api_version="2025-01-01-preview", # type: ignore
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    ))

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    generator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15", # type: ignore
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    ))

    from ragas.testset.persona import Persona

    personas = [
        Persona(
            name="curious student",
            role_description="A student who is curious about the world and wants to learn more about different cultures and languages",
        ),
    ]

    from ragas.testset.transforms.extractors.llm_based import NERExtractor
    from ragas.testset.transforms.splitters import HeadlineSplitter

    #transforms = [HeadlineSplitter(), NERExtractor()]
    transforms = [HeadlineSplitter()]

    from ragas.testset import TestsetGenerator


    personas = [
        Persona(
            name="New Joinee",
            role_description="会社についてあまり知らないので、社内のルールを確認したい。",
        ),
    ]

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas)
    #generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings) 

    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    distribution = [
         (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
         (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
    ]
    #from ragas.testset.synthesizers import default_query_distribution
    #distribution = default_query_distribution(llm=generator_llm)

    for query, _ in distribution:
        prompts = await query.adapt_prompts("japanese", llm=generator_llm)
        query.set_prompts(**prompts)

    generated_testset = generator.generate_with_langchain_docs(
        docs[:],
        testset_size=10,
        #transforms=transforms,
        query_distribution=distribution, # type: ignore
    )

    import pickle
    TEST_DATASET_FILEPATH = "generated_testset"
    with open(TEST_DATASET_FILEPATH, 'wb') as f:
            pickle.dump(generated_testset, f)

    df = generated_testset.to_pandas() # type: ignore
    df.head()

    csv_file_path = "generated_testset.csv"
    df.to_csv(csv_file_path, index=False)

asyncio.run(main())