# 🏛️ PromptCouncil - Multi-LLM Arena

Envoie un prompt à ChatGPT, Grok, Gemini et Claude en parallèle et compare les réponses côte à côte (streaming).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis éditez vos clés
streamlit run app.py
```

Les clés et les noms de modèles sont aussi modifiables dans la barre latérale.
