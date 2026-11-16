"""PromptCouncil - Multi-LLM Arena.

Envoie un même prompt à ChatGPT, Grok, Gemini et Claude en parallèle et affiche
les réponses en streaming, côte à côte.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)  # .env prioritaire sur les variables système

REQUEST_TIMEOUT_S = 120

# --------------------------------------------------------------------------- #
# Appels aux fournisseurs (générateurs : ("delta", str) puis ("usage", dict))
# --------------------------------------------------------------------------- #

Event = tuple[str, object]


EFFORT_IGNORED = "\n\n> ℹ️ Réglage d'effort non pris en charge par ce modèle : ignoré."


def _stream_chat_completions(base_url, api_key, model, system, prompt, temperature, max_tokens,
                             effort=None, max_tokens_param="max_tokens") -> Iterator[Event]:
    """API Chat Completions compatible OpenAI (OpenAI et xAI)."""
    from openai import BadRequestError, OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_S)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )
    kwargs[max_tokens_param] = max_tokens
    if effort:
        kwargs["reasoning_effort"] = effort
    effort_ignored = False
    while True:
        try:
            stream = client.chat.completions.create(**kwargs)
            break
        except BadRequestError as exc:
            msg = str(exc).lower()
            # Les modèles de raisonnement (gpt-5, o-series…) n'acceptent que la température
            # par défaut ; d'autres refusent reasoning_effort : on retire le paramètre fautif.
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
            elif "reasoning" in msg and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                effort_ignored = True
            else:
                raise
    usage = None
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield "delta", delta
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    if effort_ignored:
        yield "delta", EFFORT_IGNORED
    if usage:
        out = usage.completion_tokens or 0
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(details, "reasoning_tokens", None) or 0) if details else 0
        # OpenAI inclut la réflexion dans completion_tokens ; xAI non (elle n'apparaît que
        # dans total_tokens). On détecte le cas via le total pour compter toute la sortie facturée.
        total = getattr(usage, "total_tokens", None)
        if reasoning and total == (usage.prompt_tokens or 0) + out + reasoning:
            out += reasoning
        pdetails = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(pdetails, "cached_tokens", None) or 0) if pdetails else 0
        result = {"in": usage.prompt_tokens or 0, "out": out, "reasoning": reasoning,
                  "cached": cached}
        # xAI renvoie le coût exact facturé, en ticks (1 tick = 1e-10 $).
        ticks = getattr(usage, "cost_in_usd_ticks", None)
        if ticks is not None:
            result["cost"] = ticks / 1e10
        yield "usage", result


def stream_chatgpt(api_key, model, system, prompt, temperature, max_tokens,
                   effort=None) -> Iterator[Event]:
    yield from _stream_chat_completions(None, api_key, model, system, prompt, temperature,
                                        max_tokens, effort, max_tokens_param="max_completion_tokens")


def stream_grok(api_key, model, system, prompt, temperature, max_tokens,
                effort=None) -> Iterator[Event]:
    yield from _stream_chat_completions("https://api.x.ai/v1", api_key, model, system, prompt,
                                        temperature, max_tokens, effort)


# Gemini 2.5 règle la réflexion par budget de tokens (Gemini 3+ : thinking_level).
GEMINI_25_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def stream_gemini(api_key, model, system, prompt, temperature, max_tokens,
                  effort=None) -> Iterator[Event]:
    from google import genai
    from google.genai import errors, types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),
    )
    thinking = None
    if effort:
        thinking = (types.ThinkingConfig(thinking_budget=GEMINI_25_BUDGETS[effort])
                    if model.startswith("gemini-2.") else types.ThinkingConfig(thinking_level=effort))

    def make_config(with_thinking: bool):
        return types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=thinking if with_thinking else None,
        )

    try:
        stream = client.models.generate_content_stream(model=model, contents=prompt,
                                                       config=make_config(True))
        first = next(stream, None)
        effort_ignored = False
    except errors.ClientError as exc:
        if not thinking or "think" not in str(exc).lower():
            raise
        stream = client.models.generate_content_stream(model=model, contents=prompt,
                                                       config=make_config(False))
        first = next(stream, None)
        effort_ignored = True
    usage = None
    # Le premier morceau a été lu pour détecter un refus du réglage ; on le remet en tête.
    for chunk in itertools.chain([first] if first else [], stream):
        if chunk.text:
            yield "delta", chunk.text
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
    if effort_ignored:
        yield "delta", EFFORT_IGNORED
    if usage:
        thoughts = usage.thoughts_token_count or 0
        out = (usage.candidates_token_count or 0) + thoughts
        yield "usage", {"in": usage.prompt_token_count or 0, "out": out, "reasoning": thoughts,
                        "cached": usage.cached_content_token_count or 0}


# Modèles Claude acceptant le repli serveur en cas de refus (fallbacks="default").
CLAUDE_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
                          "claude-fable-5-2", "claude-fable-5-1"}


def stream_claude(api_key, model, system, prompt, temperature, max_tokens,
                  effort=None) -> Iterator[Event]:
    """API Messages d'Anthropic.

    La température n'est pas envoyée : les modèles Claude récents la refusent (400).
    Le raisonnement (thinking) est toujours actif sur Opus 5.5 et compte dans max_tokens.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S)
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    if model in CLAUDE_FALLBACK_MODELS:
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if effort:
        kwargs["output_config"] = {"effort": effort}

    effort_ignored = False
    try:
        with client.beta.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield "delta", text
            final = stream.get_final_message()
    except anthropic.BadRequestError as exc:
        # Modèle sans réglage d'effort (ex. Haiku 4.5) : la requête est refusée avant toute sortie.
        if "output_config" not in kwargs or "effort" not in str(exc).lower():
            raise
        kwargs.pop("output_config")
        effort_ignored = True
        with client.beta.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield "delta", text
            final = stream.get_final_message()
    if effort_ignored:
        yield "delta", EFFORT_IGNORED

    if final.stop_reason == "refusal":
        cat = getattr(final.stop_details, "category", None) if final.stop_details else None
        yield "delta", f"\n\n> ⛔ Requête refusée par Claude{f' ({cat})' if cat else ''}."
    elif final.stop_reason == "max_tokens":
        yield "delta", "\n\n> ✂️ Réponse tronquée : augmentez « Max tokens »."
    # output_tokens inclut la réflexion, mais l'API n'en donne pas le détail : reasoning=None.
    # input_tokens exclut les tokens lus depuis le cache ou écrits dans le cache : on les rajoute.
    u = final.usage
    cached = getattr(u, "cache_read_input_tokens", None) or 0
    written = getattr(u, "cache_creation_input_tokens", None) or 0
    yield "usage", {"in": u.input_tokens + cached + written, "out": u.output_tokens,
                    "reasoning": None, "cached": cached, "cache_write": written}


PROVIDERS: dict[str, dict] = {
    "chatgpt": {
        "label": "OpenAI ChatGPT",
        "badge": "linear-gradient(90deg, #0D8A6A, #10A37F)",
        "env": "OPENAI_API_KEY",
        "model": "gpt-5",  # modèle par défaut
        "cache_ratio": 0.10,  # tarif des tokens d'entrée lus depuis le cache / tarif d'entrée
        "models": {  # id -> tarif $ / 1M tokens (in, out) ; None = inconnu, à saisir
            "gpt-5": (1.25, 10.0),
            "gpt-6.1-sol": (2.0, 10.0),
            "gpt-6-sol": (2.0, 10.0),
            "gpt-6-luna": (0.10, 0.50),
            "gpt-5.6-sol": (4.0, 20.0),
            "gpt-5.6-terra": (2.0, 12.0),
            "gpt-5.6-luna": (0.20, 1.20),
            "gpt-5.5": (5.0, 30.0),
            "gpt-5-mini": (0.25, 2.0),
            "gpt-5-nano": (0.05, 0.40),
        },
        "fn": stream_chatgpt,
    },
    "grok": {
        "label": "xAI Grok",
        "badge": "linear-gradient(90deg, #111111, #555555)",
        "env": "XAI_API_KEY",
        "model": "grok-4.7",  # modèle par défaut
        "cache_ratio": 0.25,  # sert seulement si xAI ne renvoie pas le coût exact
        "models": {
            "grok-4": (3.0, 15.0),
            "grok-4.7": (2.0, 6.0),
            "grok-4.6": None,
            "grok-4.5": None,
            "grok-4.3": (1.25, 2.50),
        },
        "fn": stream_grok,
    },
    "gemini": {
        "label": "Google Gemini",
        "badge": "linear-gradient(90deg, #1A73E8, #4285F4)",
        "env": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash",  # modèle par défaut
        "cache_ratio": 0.10,
        "models": {
            "gemini-3.5-flash": (1.50, 9.0),
            "gemini-3.8-flash": None,
            "gemini-3.7-flash": None,
            "gemini-3.6-flash": None,
            "gemini-3.5-flash-lite": (0.30, 2.50),
        },
        "fn": stream_gemini,
    },
    "claude": {
        "label": "Anthropic Claude",
        "badge": "linear-gradient(90deg, #C15F3C, #D97757)",
        "env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-5-5",  # modèle par défaut
        "cache_ratio": 0.10,
        "models": {
            "claude-opus-5-5": (4.0, 20.0),
            "claude-fable-5-2": (8.0, 40.0),
            "claude-sonnet-5-5": (2.0, 10.0),
            "claude-haiku-5-5": (0.10, 0.50),
            "claude-opus-5": (5.0, 25.0),
        },
        "fn": stream_claude,
    },
}

OTHER_MODEL = "Autre…"
EFFORTS = {"Défaut du modèle": None, "Faible (low)": "low", "Moyen (medium)": "medium",
           "Élevé (high)": "high"}

# Ordre d'affichage des colonnes (et du tableau de stats).
PROVIDERS = {pid: PROVIDERS[pid] for pid in ("claude", "grok", "gemini", "chatgpt")}


# --------------------------------------------------------------------------- #
# Exécution parallèle
# --------------------------------------------------------------------------- #


def format_error(exc: Exception) -> str:
    name = type(exc).__name__
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    msg = str(exc) or name
    body = getattr(exc, "body", None)
    if isinstance(body, str) and body not in msg:  # xAI renvoie le détail dans body
        msg = f"{msg} — {body}"
    if name == "AuthenticationError" or code in (401, 403) or "api key" in msg.lower():
        return f"🔑 Clé API invalide ou non autorisée ({name}).\n\n{msg}"
    if name == "RateLimitError" or code == 429:
        return f"⏳ Limite de requêtes atteinte (rate limit).\n\n{msg}"
    if "Timeout" in name or "timed out" in msg.lower():
        return f"⌛ Délai dépassé ({REQUEST_TIMEOUT_S}s)."
    if name == "NotFoundError" or code == 404:
        return f"❓ Modèle introuvable — vérifiez son nom dans la barre latérale.\n\n{msg}"
    if "Connection" in name:
        return f"🌐 Erreur réseau ({name}).\n\n{msg}"
    return f"{name}: {msg}"


def worker(pid: str, fn: Callable[..., Iterator[Event]], kwargs: dict, q: queue.Queue) -> None:
    """Exécuté dans un thread : ne touche jamais à Streamlit, pousse tout dans la queue."""
    start = time.perf_counter()
    first = True
    try:
        for kind, payload in fn(**kwargs):
            if first and kind == "delta":
                q.put((pid, "ttft", time.perf_counter() - start))
                first = False
            q.put((pid, kind, payload))
        q.put((pid, "done", time.perf_counter() - start))
    except Exception as exc:  # noqa: BLE001 - on veut tout capturer par colonne
        q.put((pid, "error", (format_error(exc), time.perf_counter() - start)))


def new_result(pid: str, model: str, price: tuple[float, float], effort: str | None = None) -> dict:
    return {"pid": pid, "model": model, "price": price, "effort": effort, "text": "",
            "usage": None, "latency": None,
            "ttft": None, "error": None, "done": False}


def cost_usd(tokens_in: int, tokens_out: int, price: tuple[float, float]) -> float:
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000


CLAUDE_CACHE_WRITE_RATIO = 1.25  # écriture dans le cache (TTL 5 min) : 1,25 × le tarif d'entrée


def run_cost(pid: str, usage: dict, price: tuple[float, float]) -> tuple[float, bool]:
    """Coût d'un appel en $ et s'il est exact (renvoyé par l'API) ou estimé (tarifs)."""
    if usage.get("cost") is not None:
        return usage["cost"], True
    cached, written = usage.get("cached", 0), usage.get("cache_write", 0)
    normal = max(usage["in"] - cached - written, 0)
    tokens_in = (normal + cached * PROVIDERS[pid]["cache_ratio"]
                 + written * CLAUDE_CACHE_WRITE_RATIO)
    return cost_usd(tokens_in, usage["out"], price), False


# --------------------------------------------------------------------------- #
# Statistiques cumulées (persistées dans stats.json)
# --------------------------------------------------------------------------- #

STATS_FILE = Path(__file__).with_name("stats.json")


def empty_stats() -> dict:
    # ttft / ttft_n : somme et nombre des temps avant premier token (absent sur les anciens runs).
    # reasoning : tokens de réflexion (déjà inclus dans "out").
    return {"requests": 0, "errors": 0, "time": 0.0, "in": 0, "out": 0,
            "reasoning": 0, "ttft": 0.0, "ttft_n": 0, "cost": 0.0,
            "cached": 0, "exact_n": 0}  # cached : tokens d'entrée lus depuis le cache
                                         # exact_n : requêtes dont le coût vient de l'API


LEGACY_MODEL = "historique (modèles mélangés)"


def stats_key(pid: str, model: str) -> str:
    return f"{pid}|{model}"


def load_stats() -> dict[str, dict]:
    """Stats cumulées par (fournisseur, modèle) : {"pid|modèle": {provider, model, ...}}."""
    try:
        data = json.loads(STATS_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    if "version" not in data:  # ancien format : une entrée par fournisseur, modèles mélangés
        legacy = {}
        for pid, p in PROVIDERS.items():
            saved = data.get(pid)
            if not saved or not saved.get("requests"):
                continue
            s = {**empty_stats(), **saved, "provider": pid, "model": LEGACY_MODEL}
            if "cost" not in saved:  # d'avant le suivi du coût : tarif du modèle par défaut
                s["cost"] = cost_usd(s["in"], s["out"], p["models"][p["model"]])
            legacy[stats_key(pid, LEGACY_MODEL)] = s
        return legacy
    return {k: {**empty_stats(), **v} for k, v in data.get("models", {}).items()
            if v.get("provider") in PROVIDERS}


def save_stats(stats: dict[str, dict]) -> None:
    STATS_FILE.write_text(json.dumps({"version": 2, "models": stats}, indent=2))


def record_run(results: dict[str, dict]) -> None:
    """Ajoute aux stats chaque appel réellement envoyé (ceux qui ont une latence)."""
    stats = load_stats()
    for pid, res in results.items():
        if res["latency"] is None:  # clé ou modèle manquant : aucun appel
            continue
        # Une ligne par (modèle, effort) pour pouvoir comparer les niveaux d'effort.
        label = res["model"] + (f" · effort {res['effort']}" if res.get("effort") else "")
        s = stats.setdefault(stats_key(pid, label), {**empty_stats(), "provider": pid,
                                                     "model": label})
        s["requests"] += 1
        s["time"] += res["latency"]
        if res["error"]:
            s["errors"] += 1
        if res["ttft"] is not None:
            s["ttft"] += res["ttft"]
            s["ttft_n"] += 1
        if res["usage"]:
            s["in"] += res["usage"]["in"]
            s["out"] += res["usage"]["out"]
            s["reasoning"] += res["usage"].get("reasoning") or 0
            s["cached"] += res["usage"].get("cached", 0)
            cost, exact = run_cost(pid, res["usage"], res["price"])
            s["cost"] += cost
            s["exact_n"] += exact
    save_stats(stats)


def render_stats() -> None:
    import pandas as pd

    st.divider()
    head, btn = st.columns([4, 1])
    head.subheader("📊 Statistiques cumulées")
    with btn.popover("🗑️ Réinitialiser"):
        st.caption("Effacer toutes les statistiques cumulées ?")
        if st.button("Confirmer l'effacement", type="primary"):
            save_stats({})
    stats = load_stats()
    if not any(s["requests"] for s in stats.values()):
        st.caption("Aucune requête enregistrée pour l'instant.")
        return

    # Ordre : fournisseurs dans l'ordre des colonnes, puis modèles les plus utilisés.
    order = list(PROVIDERS)
    entries = sorted(stats.values(), key=lambda s: (order.index(s["provider"]), -s["requests"]))
    rows = []
    for s in entries:
        pid, n = s["provider"], s["requests"]
        if not n:
            continue
        # Claude ne détaille pas la réflexion (incluse dans « out ») : on affiche « — ».
        reasoning = s["reasoning"] if pid != "claude" else None
        rows.append({
            "Modèle": s["model"] if s["model"] != LEGACY_MODEL else f"{pid} · {LEGACY_MODEL}",
            "Fournisseur": PROVIDERS[pid]["label"],
            "Requêtes": n,
            "Erreurs": s["errors"],
            "TTFT moyen (s)": round(s["ttft"] / s["ttft_n"], 2) if s["ttft_n"] else None,
            "Temps moyen (s)": round(s["time"] / n, 2) if n else 0.0,
            "Temps total (s)": round(s["time"], 2),
            "Tokens in": s["in"],
            "dont cache": s["cached"],
            "Tokens out": s["out"],
            "dont réflexion": reasoning,
            "Débit (tok/s)": round(s["out"] / s["time"], 1) if s["time"] else 0.0,
            "Tokens total": s["in"] + s["out"],
            "Coût ($)": round(s["cost"], 4),
            "Coût / req ($)": round(s["cost"] / n, 4) if n else 0.0,
            "Source du coût": ("exact (API)" if s["exact_n"] == n
                               else "estimé" if not s["exact_n"] else "mixte"),
        })
    df = pd.DataFrame(rows).set_index("Modèle")

    total = df.sum(numeric_only=True)
    m = st.columns(4)
    m[0].metric("Requêtes", int(total["Requêtes"]))
    m[1].metric("Temps total", f"{total['Temps total (s)']:.1f} s")
    m[2].metric("Tokens total", f"{int(total['Tokens total']):,}".replace(",", " "))
    m[3].metric("Coût", f"${total['Coût ($)']:.4f}")

    st.dataframe(df, width="stretch")
    if any(s["model"] == LEGACY_MODEL for s in entries):
        st.caption("ℹ️ Les lignes « historique » regroupent les requêtes enregistrées avant le "
                   "suivi par modèle : réinitialisez pour repartir sur des stats propres.")
    st.caption(
        "TTFT = temps avant le premier token de réponse. « Tokens out » = toute la sortie "
        "facturée, réflexion comprise (« — » : Claude ne détaille pas la part de réflexion). "
        "Coût : exact quand l'API le renvoie (xAI), sinon estimé avec le tarif du modèle "
        "utilisé à chaque requête (barre latérale), tokens en cache au tarif réduit."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("⏱️ Temps moyen par modèle (s) : premier token puis reste")
        lat = pd.DataFrame({
            "Avant 1er token": df["TTFT moyen (s)"].fillna(0),
            "Génération": (df["Temps moyen (s)"] - df["TTFT moyen (s)"].fillna(0)).clip(lower=0),
        })
        st.bar_chart(lat, horizontal=True, stack=True)
    with c2:
        st.caption("🔢 Tokens par modèle (in / out)")
        st.bar_chart(df[["Tokens in", "Tokens out"]], horizontal=True, stack=True)
    with c3:
        st.caption("💰 Coût estimé par modèle ($)")
        st.bar_chart(df["Coût ($)"], horizontal=True)


# --------------------------------------------------------------------------- #
# Rendu
# --------------------------------------------------------------------------- #


def badge_html(pid: str) -> str:
    p = PROVIDERS[pid]
    return (
        f'<div style="background:{p["badge"]};color:white;padding:8px 14px;'
        f'border-radius:10px;font-weight:600;margin-bottom:10px;">{p["label"]}</div>'
    )


def render_result(container, res: dict) -> None:
    with container.container():
        st.caption(f"Modèle : `{res['model']}`")
        if res["error"]:
            st.error(res["error"])
        elif not res["done"]:
            st.markdown((res["text"] or "_En attente de la réponse…_") + " ▌")
            return
        else:
            st.markdown(res["text"] or "_(réponse vide)_")

        meta = []
        if res.get("ttft") is not None:
            meta.append(f"⚡ 1er token {res['ttft']:.2f} s")
        if res["latency"] is not None:
            meta.append(f"⏱️ {res['latency']:.2f} s")
        if res["usage"]:
            u = res["usage"]
            c = f" (dont {u['cached']} en cache)" if u.get("cached") else ""
            r = f" (dont {u['reasoning']} réflexion)" if u.get("reasoning") else ""
            meta.append(f"🔢 {u['in']} in{c} / {u['out']} out{r}")
            if res.get("price") and res.get("pid"):
                cost, exact = run_cost(res["pid"], u, res["price"])
                meta.append(f"💰 ${cost:.4f} ({'exact' if exact else 'estimé'})")
        if meta:
            st.caption(" · ".join(meta))
        if res["text"] and not res["error"]:
            with st.expander("📋 Copier la réponse"):
                st.code(res["text"], language="markdown", wrap_lines=True)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="PromptCouncil", page_icon="🏛️", layout="wide")
st.title("🏛️ PromptCouncil - Multi-LLM Arena")
st.caption("Un prompt, quatre avis : Claude, Grok, Gemini et ChatGPT répondent en parallèle.")

with st.sidebar:
    st.header("⚙️ Configuration")
    settings: dict[str, dict] = {}
    for pid, p in PROVIDERS.items():
        st.subheader(p["label"])
        enabled = st.checkbox("Activer", value=True, key=f"{pid}_enabled")
        key = st.text_input(
            p["env"],
            value=os.getenv(p["env"], ""),
            type="password",
            key=f"{pid}_key",
            help="Chargée depuis .env / l'environnement si présente ; modifiable ici.",
        )
        options = [*p["models"], OTHER_MODEL]
        choice = st.selectbox("Modèle", options, index=options.index(p["model"]),
                              key=f"{pid}_model_choice")
        if choice == OTHER_MODEL:
            model = st.text_input("Nom exact du modèle", key=f"{pid}_model_custom",
                                  placeholder=f"ex : {p['model']}")
        else:
            model = choice
        default_price = p["models"].get(model.strip())
        # Clé par modèle : changer de modèle recharge son tarif par défaut.
        with st.expander("💰 Tarif ($ / 1M tokens)", expanded=default_price is None):
            if default_price is None:
                st.caption("Tarif inconnu : renseignez-le pour estimer le coût.")
            dp = default_price or (0.0, 0.0)
            pc = st.columns(2)
            p_in = pc[0].number_input("Entrée", min_value=0.0, value=dp[0], step=0.05,
                                      format="%.2f", key=f"{pid}_{model}_price_in")
            p_out = pc[1].number_input("Sortie", min_value=0.0, value=dp[1], step=0.05,
                                       format="%.2f", key=f"{pid}_{model}_price_out")
        settings[pid] = {"enabled": enabled, "key": key.strip(), "model": model.strip(),
                         "price": (p_in, p_out)}
    st.divider()
    temperature = st.slider("Température", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=64, max_value=32000, value=4096, step=256)
    effort_label = st.selectbox(
        "Effort de réflexion", list(EFFORTS), index=0,
        help="Appliqué à tous les modèles : reasoning_effort (OpenAI, xAI), thinking_level "
             "(Gemini 3+), budget de réflexion (Gemini 2.5), effort (Claude). « Défaut » "
             "laisse chaque modèle décider. Ignoré, avec une mention, si le modèle ne le "
             "prend pas en charge.",
    )
    effort = EFFORTS[effort_label]

with st.expander("🧭 System Prompt (commun à tous les modèles)"):
    system_prompt = st.text_area("System prompt", label_visibility="collapsed", height=100,
                                 placeholder="Ex : Tu es un expert concis. Réponds en français.")

prompt = st.text_area("Votre prompt", height=160, placeholder="Posez votre question au Conseil…")
submitted = st.button("🏛️ Consulter le Conseil", type="primary")

cols = st.columns(len(PROVIDERS))
placeholders = {}
for col, pid in zip(cols, PROVIDERS):
    with col:
        st.markdown(badge_html(pid), unsafe_allow_html=True)
        placeholders[pid] = st.empty()

if submitted:
    if not prompt.strip():
        st.warning("Veuillez saisir un prompt.")
        st.stop()
    active = [pid for pid, s in settings.items() if s["enabled"]]
    if not active:
        st.warning("Activez au moins un modèle dans la barre latérale.")
        st.stop()

    results = {pid: new_result(pid, settings[pid]["model"], settings[pid]["price"], effort)
               for pid in active}
    st.session_state["results"] = results
    q: queue.Queue = queue.Queue()
    pending = set()

    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        for pid in active:
            s = settings[pid]
            if not s["key"]:
                results[pid].update(error=f"🔑 Clé `{PROVIDERS[pid]['env']}` manquante.", done=True)
                continue
            if not s["model"]:
                results[pid].update(error="❓ Nom de modèle vide.", done=True)
                continue
            kwargs = dict(api_key=s["key"], model=s["model"], system=system_prompt.strip(),
                          prompt=prompt, temperature=temperature, max_tokens=int(max_tokens),
                          effort=effort)
            pool.submit(worker, pid, PROVIDERS[pid]["fn"], kwargs, q)
            pending.add(pid)

        for pid in PROVIDERS:
            if pid in results:
                render_result(placeholders[pid], results[pid])
            else:
                placeholders[pid].info("Modèle désactivé.")

        # Boucle principale : draine la queue et rafraîchit les colonnes modifiées.
        while pending:
            try:
                items = [q.get(timeout=0.1)]
            except queue.Empty:
                continue
            while True:
                try:
                    items.append(q.get_nowait())
                except queue.Empty:
                    break
            changed = set()
            for pid, kind, payload in items:
                res = results[pid]
                if kind == "delta":
                    res["text"] += payload
                elif kind == "ttft":
                    res["ttft"] = payload
                elif kind == "usage":
                    res["usage"] = payload
                elif kind == "done":
                    res["latency"], res["done"] = payload, True
                    pending.discard(pid)
                elif kind == "error":
                    res["error"], res["latency"] = payload
                    res["done"] = True
                    pending.discard(pid)
                changed.add(pid)
            for pid in changed:
                render_result(placeholders[pid], results[pid])

    record_run(results)

elif "results" in st.session_state:
    results = st.session_state["results"]
    for pid in PROVIDERS:
        if pid in results:
            render_result(placeholders[pid], results[pid])
        else:
            placeholders[pid].info("Modèle désactivé.")
else:
    for pid in PROVIDERS:
        placeholders[pid].caption("La réponse apparaîtra ici.")

render_stats()
