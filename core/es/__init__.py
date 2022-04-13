from elasticsearch import Elasticsearch

es_client = Elasticsearch("http://192.168.1.18:8039")


# ES向量搜索
def feature_search(embedding):
    response = es_client.search(index="anime_face", body={
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": embedding
                    }
                }
            }
        }
    })
    res = []
    # 我们打印一下结果列表
    for face in response["hits"]["hits"]:
        res.append({
            "name": face["_source"]["name"],
            "score": face["_score"]
        })
    return res
