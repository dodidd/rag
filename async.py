import pickle
import tempfile

import uvicorn
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from utils.load_split import load_split
from utils import rerank
import json

# 全局变量，避免每次请求都重新加载
loaded_embeddings_model = None
loaded_faiss_db = None
bm25_retriever = None
ensemble_retriever = None
PDF_PATH = r"./金融数据集-报表"
EMBEDDING_MODEL_NAME_OR_PATH = ''
FAISS_DB_PATH = r"./faiss_index_bge_m3"
METADATA_FILE_NAME = "documents_metadata.json"
BM25_INDEX_PATH = r"./bm25_index"

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate # 导入 ChatPromptTemplate

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek API 密钥和基础 URL
deepseek_api_key = os.getenv("SILICONFLOW_API_KEY")
deepseek_api_base = "https://api.siliconflow.cn/v1"

# 初始化 LangChain 的 ChatOpenAI 模型，用于调用 DeepSeek API
llm = ChatOpenAI(
    model_name="deepseek-ai/DeepSeek-V3.1", # 也可以尝试 "deepseek-coder"
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    temperature=0.1 # 降低温度，让模型生成更确定、更少发散的代码
)

chainA_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你要依据输入的上下文信息和问题进行总结"),
        ("human", "请你依据问题简单解释一下：{input}"),
    ]
)
from langchain.chains import LLMChain # LLMChain类在0.1.17版本中已被弃用，并将在1.0版本中移除

chainA_chains = LLMChain(llm=llm,
                         prompt=chainA_template,
                         verbose=True
                        )

chainC_template = ChatPromptTemplate.from_messages(
    [

        ("system", "你非常善于提取文本中的重要信息，并做出一段话或分点总结"),
        ("human", "这是针对一个提问完整的解释说明内容：{input}"),
    ]
)

chainC_chains = LLMChain(llm=llm,
                         prompt=chainC_template,
                         verbose=True
                        )

# 导入SimpleSequentialChain
from langchain.chains import SimpleSequentialChain

# 在chains参数中，按顺序传入LLMChain A 和LLMChain B,c
full_chain = SimpleSequentialChain(chains=[chainA_chains,chainC_chains], verbose=True)



async def initialize_retrievers():
    """初始化检索器，只需要执行一次"""
    global loaded_embeddings_model, loaded_faiss_db, bm25_retriever, ensemble_retriever

    document_chunks = []
    if os.path.exists(FAISS_DB_PATH + '/' + METADATA_FILE_NAME):
        with open(FAISS_DB_PATH + '/' + METADATA_FILE_NAME, 'r', encoding='utf-8') as f:
            raw_documents = json.load(f)


        for doc_dict in raw_documents:
            page_content = doc_dict.get('page_content_preview', '')

            # 构建元数据，包含所有相关信息
            metadata = {
                'chunk_id': doc_dict.get('chunk_id', ''),
                'source': doc_dict.get('source', ''),
                'page': doc_dict.get('page', 0),
                'start_index': doc_dict.get('start_index', 0)
            }
            document_chunks.append(Document(
                page_content=page_content,
                metadata=metadata))
    else:
        document_chunks = load_split(PDF_PATH)

    # print(f"成功转换 {len(document_chunks)} 个文档块")
    # print(document_chunks[0])
    if loaded_embeddings_model is None:
        # 1. 加载嵌入模型和FAISS数据库
        loaded_embeddings_model = rerank.get_embeddings_model(rerank.EMBEDDING_MODEL_NAME_OR_PATH)
        loaded_faiss_db = FAISS.load_local(rerank.FAISS_DB_PATH, loaded_embeddings_model,
                                           allow_dangerous_deserialization=True)
        print(f"已从 {rerank.FAISS_DB_PATH} 加载 FAISS 数据库。")

        # 2. 初始化 BM25 关键词检索器
        #若无，初始化，若有则加载
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, 'rb') as f:
                bm25_data = pickle.load(f)
                bm25_retriever = bm25_data['retriever']
        else:
            bm25_retriever = BM25Retriever.from_documents(document_chunks)
            bm25_retriever.k = 6  # 设置关键词检索召回数量
            # 保存到文件
            with open(BM25_INDEX_PATH, 'wb') as f:
                pickle.dump({'retriever': bm25_retriever}, f)

        # 3. 初始化 FAISS 向量检索器
        vector_retriever = loaded_faiss_db.as_retriever(search_kwargs={"k": bm25_retriever.k})  # 设置向量检索召回数量

        # 4. 组合成 EnsembleRetriever 混合检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]  # 可以调整两种检索器的权重
        )
        print("BM25 和 FAISS 检索器已组合成 EnsembleRetriever。")


# 异步处理函数
async def homepage(request):
    return JSONResponse({"message": "Hello, Starlette!"})


async def rag_query(request):
    """处理RAG查询请求"""
    try:
        # 初始化检索器（如果尚未初始化）
        await initialize_retrievers()

        # 获取请求体中的JSON数据
        body = await request.json()
        user_query = body.get("question",'None')

        if not user_query:
            return JSONResponse({"error": "缺少question参数"}, status_code=400)

        # 执行检索和重排序
        initial_retrieved_docs = ensemble_retriever.invoke(user_query)
        top_reranked_docs = rerank.rerank_documents_siliconflow(user_query, initial_retrieved_docs, top_n=6)
        context_text = "\n\n".join([doc.page_content for doc in top_reranked_docs ])+"question:"+user_query
        answer = full_chain.invoke({"input": context_text })
        print("top_reranked_docs :",top_reranked_docs )
        # 格式化响应
        response_data = {
            "question": user_query,
            "answer": answer['output'],
            "success": True,
            "retrieved_count": len(initial_retrieved_docs),
            "content": [
                {
                    "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "score": getattr(doc, 'score', 0.0) if hasattr(doc, 'score') else 0.0
                }
                for doc in top_reranked_docs
            ]
        }

        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"error": f"处理请求时出错: {str(e)}"}, status_code=500)


def rebuild_bm25_index(new_chunks, bm25_index_path):
    global all_documents
    all_documents.extend([chunk.page_content for chunk in new_chunks])

    pass
# 添加文件上传接口

async def upload_file(request):
    try:
        form = await request.form()
        file = form["file"]  # 获取文件

        # 创建临时文件目录（如果不存在）
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        # 生成临时文件路径
        temp_path = os.path.join(temp_dir, file.filename)

        # 1. 首先保存文件到临时路径
        file_content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(file_content)

        print(f"文件已保存到: {temp_path}")

        # 2. 然后处理文件
        # chunks = rerank.load_and_split_pdf(temp_path)
        chunks = load_split(temp_path)

        if not chunks:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "无法从PDF中提取文本内容"}
            )

        print(f"成功处理文档，生成 {len(chunks)} 个文本块")

        # 3. 处理完成后清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"临时文件已清理: {temp_path}")

        # 4. 更新FAISS数据库
        embedding_model = rerank.get_embeddings_model('')
        if os.path.exists(FAISS_DB_PATH):
            # 如果数据库已存在，添加新内容
            rerank.add_to_faiss(
                new_chunks=chunks,
                embedding_model=embedding_model,  # 确保这个变量已定义
                faiss_db_path=FAISS_DB_PATH,

            )
            print("已更新现有FAISS数据库")
        else:
            # 如果数据库不存在，创建新的
            rerank.create_and_save_faiss_db(
                chunks,
                embedding_model,
                FAISS_DB_PATH
            )
            rerank.create_and_save_metadata(
                chunks,
                FAISS_DB_PATH,
                METADATA_FILE_NAME
            )
            print("已创建新的FAISS数据库")

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks_count": len(chunks)
        })

    except Exception as e:
        # 异常处理：确保清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"发生错误，已清理临时文件: {temp_path}")

        print(f"文件上传处理失败: {e}")
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"文件处理失败: {str(e)}"}
        )

# 路由配置
routes = [
    Route("/", endpoint=homepage, methods=["GET"]),
    Route("/upload", endpoint=upload_file, methods=["POST"]),
    Route("/rag_query", endpoint=rag_query, methods=["POST"]),  # 修正：使用POST方法
]

# 创建应用
app = Starlette(routes=routes)

if __name__ == "__main__":
    print("启动RAG检索服务...")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5000,
        reload=False  # 开发时启用热重载
    )