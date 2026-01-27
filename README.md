# SAST_Automation


llm_verifier
.env 구성해야됨
```
GOOGLE_API_KEY={your gemini key}
LLM_MODEL={your llm model}

```

현재 종속성(일단 llm 코드것만)
python,node,npm 
  Python 3.12 -> 최신 버전 사용시 langgraph error
  node v24.12.0
  npm 11.6.2

requirements.txt
   langgraph
   langchain-core
   langchain-openai
   langchain-google-genai
   numpy
   sentence-transformers

   pip install -r requirements.txt 로 requirements 한번에 설치
