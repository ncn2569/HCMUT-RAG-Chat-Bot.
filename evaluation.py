import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv("config/.env")
hf_home = os.getenv("HF_HOME")
if hf_home is not None:
    os.environ["HF_HOME"] = hf_home
from datasets import Dataset
from rag.embedding.embed import load_embedder
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._answer_relevance import ResponseRelevancy
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import RunConfig
from rag.core.container import container
from rag.pipeline import RAGPipeline
import time

pipeline = RAGPipeline(container)

eval_embed = GoogleGenerativeAIEmbeddings(
    model=os.getenv("model_embedding_name"), google_api_key=os.getenv("API_KEY")
)
client_rag = load_embedder()  # load client thì đúng hơn
llm = llm_factory(
    model="gemini-3.1-flash-lite-preview", provider="google", client=client_rag
)
metrics_list = [
    LLMContextPrecisionWithReference(),
    ResponseRelevancy(embeddings=eval_embed),
    LLMContextRecall(),
    Faithfulness(),
]
run_config = RunConfig(max_workers=1, max_retries=30, max_wait=120)


def eval():
    print("starting...")
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    data = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    for i, item in enumerate(test_data):
        question = item["question"]
        truth = item["ground_truth"]

        answer, context = pipeline.query_with_contexts(question)

        data["user_input"].append(question)
        data["reference"].append(truth)
        data["retrieved_contexts"].append(context)
        data["response"].append(answer)
        if i > 0 and i % 5 == 0:
            time.sleep(61)

    dataset = Dataset.from_dict(data)
    final_results = []

    # RPM của gemini 3.1 flash (không được overaload nặng quá tính vừa đủ sẽ bị vượt TPM liên tục, lỗi của ragas)
    for i in range(0, len(dataset), 2):
        chunk = dataset.select(range(i, min(i + 2, len(dataset))))
        result = evaluate(
            dataset=chunk,
            metrics=metrics_list,
            llm=llm,
            embeddings=eval_embed,
            raise_exceptions=False,
            show_progress=True,
            run_config=run_config,
        )
        print(f"Đang ở câu {i + 1}")
        final_results.append(result.to_pandas())
        if (i + 1) < len(dataset):
            time.sleep(65)  # free thì phải chịu RPM 60 để 120s cho chắc

    df = pd.concat(final_results, ignore_index=True)
    drop_columns = ["user_input"]
    df_final = df.drop(columns=drop_columns, errors="ignore")
    df_final.to_csv("report6.csv", encoding="utf-8")
    print("Done")


eval()
