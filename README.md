# LLM 학습 프로젝트

> LLM(Large Language Model), Transformer, Vector Database, RAG를 학습하기 위한 Jupyter Notebook 기반 실습 프로젝트입니다.

## 📚 프로젝트 소개

이 프로젝트는 최신 LLM 기술 스택을 단계별로 학습할 수 있도록 구성된 교육용 자료입니다. Transformer 아키텍처의 기초부터 시작하여 Vector Database, 그리고 실전 RAG 시스템 구축까지 다룹니다.

## 🗂️ 프로젝트 구조

```
LLM/
├── 01_Transformer/           # Transformer 아키텍처 학습
│   ├── 01_transformer_by_pytorch.ipynb
│   ├── 02_BERT_transfer_learning.ipynb
│   └── 03_huggingface_pipeline.ipynb
│
├── 02_VectorDB/              # 벡터 데이터베이스 실습
│   ├── 01_chroma_db.ipynb
│   └── 02_faiss_db.ipynb
│
├── 03_RAG/                   # RAG 시스템 구현
│   ├── 00_rag_workflow.ipynb
│   ├── 01_retrieval.ipynb
│   ├── 02_retriever.ipynb
│   └── images/
│
├── data/                     # 데이터 파일
│   └── tmdb_5000_movies.csv
│
├── db/                       # 벡터 DB 저장소
│   └── faiss_vector_store/
│
├── requirements.txt          # 패키지 의존성
└── README.md
```

## 📖 학습 내용

### 1️⃣ Transformer (01_Transformer/)

Transformer 아키텍처의 기초와 전이 학습을 실습합니다.

- **01_transformer_by_pytorch.ipynb**
  - PyTorch를 활용한 Transformer 모델 구현
  - Attention 메커니즘 이해
  - Encoder-Decoder 구조 학습

- **02_BERT_transfer_learning.ipynb**
  - BERT 모델을 활용한 전이 학습
  - Fine-tuning 기법
  - 텍스트 분류 태스크 실습

- **03_huggingface_pipeline.ipynb**
  - Hugging Face Transformers 라이브러리 활용
  - Pre-trained 모델 사용법
  - Pipeline을 통한 간편한 추론

### 2️⃣ Vector Database (02_VectorDB/)

임베딩 벡터를 저장하고 검색하는 Vector Database를 학습합니다.

- **01_chroma_db.ipynb**
  - ChromaDB 기본 사용법
  - 벡터 임베딩 저장 및 검색
  - 유사도 기반 검색 (Similarity Search)
  - 메타데이터 필터링

- **02_faiss_db.ipynb**
  - Facebook AI Similarity Search (FAISS) 사용법
  - 대용량 벡터 검색 최적화
  - 인덱스 타입별 특징
  - 성능 비교 및 벤치마킹

### 3️⃣ RAG (Retrieval-Augmented Generation) (03_RAG/)

검색 증강 생성(RAG) 시스템을 구축하고 실습합니다.

- **00_rag_workflow.ipynb**
  - RAG 개념 및 워크플로우 이해
  - RAG의 구성 요소 (Vector DB + LLM + 검색 엔진)
  - Document Loading & Splitting
  - 실전 활용 사례

- **01_retrieval.ipynb**
  - 문서 검색(Retrieval) 구현
  - 임베딩 모델 활용
  - 검색 쿼리 최적화

- **02_retriever.ipynb**
  - Retriever 패턴 구현
  - LangChain Retriever 활용
  - 검색 결과 후처리

## 🛠️ 기술 스택

- **언어**: Python 3.x
- **프레임워크**:
  - PyTorch
  - Transformers (Hugging Face)
  - LangChain 1.0.x
- **Vector DB**:
  - ChromaDB
  - FAISS
- **주요 라이브러리**:
  - `datasets` - 데이터셋 로드
  - `sentence-transformers` - 임베딩 생성
  - `langchain` - RAG 파이프라인 구축
  - `jupyter` - 노트북 환경

## 🚀 시작하기

### 1. 환경 설정

Python 3.8 이상이 필요합니다.

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. Jupyter Notebook 실행

```bash
jupyter notebook
```

또는 VS Code에서 `.ipynb` 파일을 직접 열어 실행할 수 있습니다.

### 3. 학습 순서

1. **Transformer 기초** → `01_Transformer/` 폴더의 노트북부터 시작
2. **Vector Database** → `02_VectorDB/` 폴더에서 벡터 검색 학습
3. **RAG 시스템** → `03_RAG/` 폴더에서 통합 시스템 구축

## 📊 데이터

- `data/tmdb_5000_movies.csv`: TMDB 영화 데이터셋 (5,000개 영화 정보)
  - 실습용 샘플 데이터로 활용

## 🔑 주요 개념

### RAG (Retrieval-Augmented Generation)

RAG는 Vector Database와 LLM을 결합하여 검색된 데이터를 활용해 문맥 기반 응답을 생성하는 기술입니다.

**RAG Workflow:**
1. 사용자 질문 입력
2. Embedding 모델을 사용해 질문을 벡터화
3. Vector DB에서 유사한 문서 검색
4. 검색된 문서를 LLM이 응답 생성에 활용
5. 사용자에게 응답 반환

**장점:**
- ✅ 최신 정보 제공 (모델 학습 이후 데이터도 활용 가능)
- ✅ 정확성 향상 (검색 데이터 기반으로 환각 현상 감소)
- ✅ 유연성 (다양한 도메인 데이터 활용 가능)

### Vector Database

고차원 벡터 데이터를 효율적으로 저장하고 유사도 기반 검색을 수행하는 데이터베이스입니다.

- **ChromaDB**: 간편한 설정과 사용, 프로토타입에 적합
- **FAISS**: 대용량 데이터 처리에 최적화, 프로덕션 환경에 적합

## 📝 주의사항

- 일부 노트북은 GPU 환경에서 실행하는 것을 권장합니다
- OpenAI API 키가 필요한 경우, `.env` 파일에 설정하세요
- LangChain 1.0.x 버전 기준으로 작성되었습니다
 

---

**Last Updated**: November 6, 2025