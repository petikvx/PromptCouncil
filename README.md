# 🏛️ PromptCouncil - Multi-LLM Arena

> Un prompt, quatre avis : **Claude**, **Grok**, **Gemini** et **ChatGPT** répondent en parallèle.

PromptCouncil est une petite application [Streamlit](https://streamlit.io) qui envoie le même prompt à plusieurs grands modèles de langage en même temps. Les réponses s'affichent côte à côte, en streaming, ce qui permet de comparer leur qualité, leur vitesse et leur coût.

## ✨ Fonctionnalités

- **Requêtes en parallèle** vers 4 fournisseurs (Anthropic, xAI, Google, OpenAI), avec affichage en streaming.
- **Configuration dans la barre latérale** : activer ou désactiver chaque modèle, choisir le modèle (ou saisir un nom exact via « Autre… »), modifier les clés API et les tarifs.
- **Réglages communs** : system prompt, température, max tokens et niveau d'effort de raisonnement (low / medium / high). Le niveau d'effort est ignoré, avec un avertissement, par les modèles qui ne le prennent pas en charge.
- **Métriques par réponse** : temps jusqu'au premier token, durée totale, tokens en entrée et en sortie (raisonnement compris), coût estimé en $ (lecture et écriture de cache comprises).
- **Statistiques cumulées**, stockées dans `stats.json` : tableau et graphiques des latences, des tokens et des coûts par modèle, avec un bouton de remise à zéro.
- **Copie facile** de chaque réponse au format Markdown.

## 🤖 Modèles par défaut

| Fournisseur | Variable d'environnement | Modèle par défaut |
|---|---|---|
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| xAI Grok | `XAI_API_KEY` | `grok-4.7` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` |
| OpenAI ChatGPT | `OPENAI_API_KEY` | `gpt-5` |

La liste des modèles proposés et leurs tarifs ($ / 1M tokens) se trouvent dans le dictionnaire `PROVIDERS` de `app.py`.

## 🚀 Installation

Prérequis : Python 3.10+ et au moins une clé API.

```bash
git clone https://github.com/petikvx/PromptCouncil.git
cd PromptCouncil
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis renseignez vos clés
```

Seules les clés des fournisseurs que vous utilisez sont nécessaires. Il suffit de désactiver les autres dans la barre latérale.

## ▶️ Lancement

**Au premier plan :**

```bash
streamlit run app.py
```

**En arrière-plan** avec le script `council.sh` :

```bash
./council.sh start     # démarre sur http://localhost:8501
./council.sh status    # indique si l'application tourne
./council.sh restart
./council.sh stop
```

Variables optionnelles :

| Variable | Défaut | Rôle |
|---|---|---|
| `PORT` | `8501` | Port d'écoute |
| `ADDRESS` | `localhost` | `0.0.0.0` pour exposer l'application sur le réseau local |

Exemple : `PORT=9000 ADDRESS=0.0.0.0 ./council.sh start`. Les logs sont écrits dans `council.log`.

> ⚠️ Avec `ADDRESS=0.0.0.0`, toute personne du réseau peut utiliser vos clés API. L'application n'a pas d'authentification.

## 📁 Structure

```
app.py             # application Streamlit (fournisseurs, streaming, coûts, stats)
council.sh         # démarrage / arrêt en arrière-plan
requirements.txt   # dépendances Python
.env.example       # modèle de fichier de clés
stats.json         # statistiques cumulées (généré, ignoré par git)
```

## 🔒 Sécurité

- `.env` est exclu du dépôt par le `.gitignore`. Ne commitez jamais vos clés.
- Les coûts affichés sont des **estimations** calculées à partir des tarifs configurés. Seule la facturation de chaque fournisseur fait foi.
