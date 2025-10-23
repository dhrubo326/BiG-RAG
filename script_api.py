import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import faiss
from FlagEmbedding import FlagAutoModel
from typing import List
import argparse
from bigrag import BiGRAG, QueryParam
import asyncio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2WikiMultiHopQA')
args = parser.parse_args()
data_source = args.data_source

# 加载 FAISS 索引和 FlagEmbedding 模型
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices="cpu",
)

# 加载 FAISS 索引和 FlagEmbedding 模型
print(f"[DEBUG] LOADING EMBEDDINGS")
index_entity = faiss.read_index(f"expr/{data_source}/index_entity.bin")
corpus_entity = []
with open(f"expr/{data_source}/kv_store_entities.json") as f:
    entities = json.load(f)
    for item in entities:
        corpus_entity.append(entities[item]['entity_name'])
print("[DEBUG] EMBEDDINGS LOADED")

# 加载 FAISS 索引和 FlagEmbedding 模型
print(f"[DEBUG] LOADING EMBEDDINGS")
index_bipartite_edge = faiss.read_index(f"expr/{data_source}/index_bipartite_edge.bin")
corpus_bipartite_edge = []
with open(f"expr/{data_source}/kv_store_bipartite_edges.json") as f:
    bipartite_edges = json.load(f)
    for item in bipartite_edges:
        corpus_bipartite_edge.append(bipartite_edges[item]['content'])
print("[DEBUG] EMBEDDINGS LOADED")

rag = BiGRAG(
    working_dir=f"expr/{data_source}",
)

async def process_query(query_text, rag_instance, entity_match, bipartite_edge_match):
    result = await rag_instance.aquery(query_text, param=QueryParam(only_need_context=True, top_k=10), entity_match=entity_match, bipartite_edge_match=bipartite_edge_match)
    return {"query": query_text, "result": result}

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def _format_results(results: List, corpus) -> str:
    results_list = []
    
    for i, result in enumerate(results):
        results_list.append(corpus[result])
    
    return results_list

def queries_to_results(queries: List[str]) -> List[str]:
    
    embeddings = model.encode_queries(queries)
    _, ids = index_entity.search(embeddings, 5)  # 每个查询返回 5 个结果
    entity_match = {queries[i]:_format_results(ids[i], corpus_entity) for i in range(len(ids))}
    _, ids = index_bipartite_edge.search(embeddings, 5)  # 每个查询返回 5 个结果
    bipartite_edge_match = {queries[i]:_format_results(ids[i], corpus_bipartite_edge) for i in range(len(ids))}

    results = []
    loop = always_get_an_event_loop()
    for query_text in tqdm(queries, desc="Processing queries", unit="query"):
        result = loop.run_until_complete(
            process_query(query_text, rag, entity_match[query_text], bipartite_edge_match[query_text])
        )
        results.append(json.dumps({"results": result["result"]}))
    return results
########### PREDEFINE ############

# 创建 FastAPI 实例
app = FastAPI(title="Search API", description="An API for document retrieval using FAISS and FlagEmbedding.")

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    results_str = queries_to_results(request.queries)
    return results_str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)