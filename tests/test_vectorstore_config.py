import pytest

from src.retrievers.vectorstore import load_index, vectorstore_backend


def test_vectorstore_defaults_to_offline_faiss(monkeypatch):
    monkeypatch.delenv("VECTORSTORE_BACKEND", raising=False)

    assert vectorstore_backend() == "faiss"


def test_vectorstore_accepts_pinecone(monkeypatch):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "pinecone")

    assert vectorstore_backend() == "pinecone"


def test_vectorstore_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "weaviate")

    with pytest.raises(ValueError, match="Unsupported VECTORSTORE_BACKEND"):
        vectorstore_backend()


def test_local_faiss_pickle_load_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "faiss")
    monkeypatch.delenv("HLM_TRUST_LOCAL_FAISS_INDEX", raising=False)

    with pytest.raises(RuntimeError, match="HLM_TRUST_LOCAL_FAISS_INDEX"):
        load_index(index_dir=tmp_path)
