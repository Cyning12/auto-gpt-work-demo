from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from Models.Factory import ChatModelFactory, EmbeddingModelFactory


class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyMuPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")


def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]


def load_docs(filename: str) -> List[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split()
    return pages


def ask_docment(
        filename: str,
        query: str,
) -> str:
    """根据一个PDF文档的内容，回答一个问题"""

    raw_docs = load_docs(filename)
    if len(raw_docs) == 0:
        return "抱歉，文档内容为空"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "无法读取文档内容"

    db = Chroma.from_documents(documents, EmbeddingModelFactory.get_default_model())
    hits = db.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in hits])

    prompt = PromptTemplate.from_template(
        "你是一个严谨的问答助手。仅基于给定的【参考资料】回答问题；如果资料不足以回答，请明确说“不知道”。\n"
        "【参考资料】\n{context}\n\n"
        "【问题】\n{query}\n\n"
        "【回答要求】\n- 用中文回答\n- 尽量简洁\n"
    )
    chain = prompt | ChatModelFactory.get_default_model() | StrOutputParser()
    return chain.invoke({"context": context, "query": query})


if __name__ == "__main__":
    filename = "../data/2023年10月份销售计划.docx"
    query = "销售额达标的标准是多少？"
    response = ask_docment(filename, query)
    print(response)
