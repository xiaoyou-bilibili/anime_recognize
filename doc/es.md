## 索引创建
```bash
PUT anime_face
{
  "mappings": {
    "properties": {
      "id" : {
        "type" : "keyword"
      },
       "name" : {
        "type" : "keyword"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 512
      }
    }
  }
}
```