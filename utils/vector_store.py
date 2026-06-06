from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer

from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):

    def __init__(self):

        self.model = SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

    def embed_documents(self, texts):

        return self.model.encode(texts).tolist()

    def embed_query(self, text):

        return self.model.encode([text])[0].tolist()


def create_vector_store(text):

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_text(text)

    embeddings = CustomEmbeddings()

    vectorstore = FAISS.from_texts(
        docs,
        embeddings
    )

    return vectorstore