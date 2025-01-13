from fastapi import FastAPI
from pydantic import BaseModel
from text2sql_pipeline import *
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Địa chỉ frontend
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, ...)
    allow_headers=["*"],  # Cho phép tất cả các headers
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else: raise Exception("CUDA not available!")

# Define the expected structure of the input JSON
class QueryRequest(BaseModel):
    query: str

# Define the structure of the output JSON
# class QueryResponse(BaseModel):
#     response: str
#     references: []

@app.post("/query_text2sql/")
async def predict(request: QueryRequest):
    message = request.query
    try:
        answer,sql,result,CoT = query(message)
        # Construct the response
        if CoT == '':
            return {"bot_response" : answer, 
                    "bot_refer": [{"title" : "translate query" , "content" : sql},
                                  {"title" : "result query" , "content" : result}]}
        else:
            return {"bot_response" : answer, 
                    "bot_refer": [{"title" : "translate query" , "content" : sql},
                                  {"title" : "CoT" , "content" : CoT},
                                  {"title" : "result query" , "content" : result}]}
    except:
        return {"bot_response" : "Không tìm kiếm được thông tin!!"}
# Optional: a simple root path to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Text2sql Agent"}