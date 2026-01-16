import os
import sys
import time
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# This is where we will store the database
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    start_time = time.time()
    
    # 1. Check if PDF exists
    if not os.path.exists("data.pdf"):
        print(" Error: I cannot find 'data.pdf'. Did you rename it?")
        return

    print("📄 Loading the Constitution of Pakistan...")
    sys.stdout.flush()
    load_start = time.time()
    loader = PyPDFLoader("data.pdf")
    documents = loader.load()
    load_time = time.time() - load_start
    print(f"   - Found {len(documents)} pages. ✓ ({load_time:.2f}s)")
    sys.stdout.flush()

    # 2. Chop it into small pieces (Chunks)
    # We chop it because AI can't read the whole book at once.
    print("✂️  Splitting into chunks...")
    sys.stdout.flush()
    split_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    split_time = time.time() - split_start
    print(f"   - Split into {len(texts)} chunks. ✓ ({split_time:.2f}s)")
    sys.stdout.flush()

    # 3. Turn Text into Numbers (Embeddings)
    print("🧠 Converting text to numbers (Vectors)...")
    print("   - Loading model (this may take a moment on first run)...")
    sys.stdout.flush()
    embed_start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    embed_init_time = time.time() - embed_start
    print(f"   - Model loaded. ✓ ({embed_init_time:.2f}s)")
    sys.stdout.flush()
    
    print("   - Creating embeddings...")
    sys.stdout.flush()
    embed_create_start = time.time()
    db = FAISS.from_documents(texts, embeddings)
    embed_create_time = time.time() - embed_create_start
    print(f"   - Embeddings created. ✓ ({embed_create_time:.2f}s)")
    sys.stdout.flush()

    # 4. Save to a local folder
    print("💾 Saving database to folder...")
    sys.stdout.flush()
    save_start = time.time()
    db.save_local(DB_FAISS_PATH)
    save_time = time.time() - save_start
    print(f"   - Database saved. ✓ ({save_time:.2f}s)")
    sys.stdout.flush()
    
    total_time = time.time() - start_time
    print(f"\n✅ SUCCESS! Vector Database is ready.")
    print(f"   Total time: {total_time:.2f}s")

if __name__ == "__main__":
    create_vector_db()