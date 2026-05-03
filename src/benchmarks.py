from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    title: str
    mode: str
    system: str
    prompt: str
    language: str = "en"
    difficulty: str = "medium"
    focus: str = ""
    expected_output: str = ""
    tools: list[dict[str, Any]] | None = None
    expected_tool: str | None = None
    requires_vision: bool = False


LANGUAGE_ORDER = ["en", "fr"]

CATEGORY_ORDER = [
    "summarization",
    "structured_extraction",
    "classification",
    "reasoning",
    "data_analysis",
    "instruction_following",
    "translation",
    "code_generation",
    "tool_calling",
    "vision",
]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total",
            "description": "Compute a checkout total from item prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "number"}},
                    "tax_rate": {"type": "number"},
                },
                "required": ["items", "tax_rate"],
            },
        },
    },
]


def _case(
    *,
    id: str,
    category: str,
    title: str,
    mode: str,
    system: str,
    prompt: str,
    language: str,
    difficulty: str,
    focus: str,
    expected_output: str,
    tools: list[dict[str, Any]] | None = None,
    expected_tool: str | None = None,
    requires_vision: bool = False,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=id,
        category=category,
        title=title,
        mode=mode,
        system=system,
        prompt=prompt,
        language=language,
        difficulty=difficulty,
        focus=focus,
        expected_output=expected_output,
        tools=copy.deepcopy(tools),
        expected_tool=expected_tool,
        requires_vision=requires_vision,
    )


BENCHMARK_CASES: list[BenchmarkCase] = [
    _case(
        id="sum-release-en",
        category="summarization",
        title="Release-note summary",
        mode="generate",
        language="en",
        difficulty="easy",
        focus="coverage, brevity, factual accuracy",
        expected_output="3 bullets plus one takeaway",
        system="Summarize clearly and preserve key facts.",
        prompt=(
            "Summarize this release note in 3 bullets and one short takeaway.\n\n"
            "Version 3.2 improves startup time by 18%, reduces memory usage in "
            "the image pipeline, fixes a race condition during sync, and adds a "
            "new admin audit export. A legacy CSV import path is deprecated and "
            "will be removed next quarter."
        ),
    ),
    _case(
        id="sum-release-fr",
        category="summarization",
        title="Résumé de note de version",
        mode="generate",
        language="fr",
        difficulty="easy",
        focus="couverture, concision, exactitude factuelle",
        expected_output="3 points et une conclusion courte",
        system="Résumez clairement en conservant les faits importants.",
        prompt=(
            "Résumez cette note de version en 3 points et une conclusion courte.\n\n"
            "La version 3.2 améliore le temps de démarrage de 18 %, réduit "
            "l'utilisation mémoire dans le pipeline d'images, corrige une condition "
            "de concurrence pendant la synchronisation et ajoute un nouvel export "
            "d'audit administrateur. Un ancien parcours d'import CSV est déprécié "
            "et sera supprimé au prochain trimestre."
        ),
    ),
    _case(
        id="sum-meeting-en",
        category="summarization",
        title="Meeting summary",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="structure, priorities, named facts",
        expected_output="decisions, risks, and next steps",
        system="Write concise structured summaries for technical stakeholders.",
        prompt=(
            "Summarize this meeting transcript into decisions, risks, and next steps.\n\n"
            "The team agreed to keep local inference because cloud costs are too "
            "high for development. Benchmarks showed Qwen was faster on code tasks, "
            "but Gemma produced more consistent summaries. Priya noted prompt "
            "templates were drifting between team members. Marcos raised concerns "
            "about tool-calling regressions after quantization, especially nested "
            "JSON arguments."
        ),
    ),
    _case(
        id="sum-meeting-fr",
        category="summarization",
        title="Résumé de réunion",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="structure, priorités, faits nommés",
        expected_output="décisions, risques et prochaines étapes",
        system="Rédigez des résumés structurés et concis pour des interlocuteurs techniques.",
        prompt=(
            "Résumez cette transcription de réunion en décisions, risques et prochaines étapes.\n\n"
            "L'équipe a décidé de conserver l'inférence locale car les coûts cloud "
            "sont trop élevés pour le développement. Les benchmarks ont montré que "
            "Qwen était plus rapide sur les tâches de code, mais que Gemma produisait "
            "des résumés plus constants. Priya a indiqué que les modèles de prompts "
            "divergeaient entre les membres de l'équipe. Marcos a signalé des risques "
            "de régression sur les appels d'outils après quantification, surtout avec "
            "les arguments JSON imbriqués."
        ),
    ),
    _case(
        id="extract-ticket-json-en",
        category="structured_extraction",
        title="Support ticket JSON",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="schema adherence, field selection, compact output",
        expected_output="valid JSON with 5 requested keys",
        system="Return only valid compact JSON matching the requested schema.",
        prompt=(
            "Extract this request as JSON with keys `customer`, `product`, "
            "`severity`, `summary`, and `requested_action`.\n\n"
            "Hi, this is Amira from Northwind. Since yesterday's update, our invoice "
            "export in LedgerPro fails for orders above 500 lines. Finance is blocked "
            "for month-end close, so please escalate this as urgent and tell us how "
            "to roll back safely."
        ),
    ),
    _case(
        id="extract-ticket-json-fr",
        category="structured_extraction",
        title="Ticket support en JSON",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="respect du schéma, choix des champs, sortie compacte",
        expected_output="JSON valide avec les 5 clés demandées",
        system="Retournez uniquement un JSON compact et valide qui respecte le schéma demandé.",
        prompt=(
            "Extrayez cette demande en JSON avec les clés `customer`, `product`, "
            "`severity`, `summary` et `requested_action`.\n\n"
            "Bonjour, je suis Amira de Northwind. Depuis la mise à jour d'hier, "
            "notre export de factures dans LedgerPro échoue pour les commandes de "
            "plus de 500 lignes. La finance est bloquée pour la clôture mensuelle, "
            "donc merci d'escalader cela en urgence et de nous expliquer comment "
            "revenir en arrière sans risque."
        ),
    ),
    _case(
        id="classify-support-en",
        category="classification",
        title="Support routing classification",
        mode="generate",
        language="en",
        difficulty="easy",
        focus="label discipline, confidence, short rationale",
        expected_output="one allowed label, confidence, and one reason",
        system="Classify with exactly one allowed label and do not invent labels.",
        prompt=(
            "Classify the message as exactly one of: billing, bug, feature, sales, other. "
            "Return `label`, `confidence`, and `reason`.\n\n"
            "Message: We upgraded to the Pro plan this morning, but the export button "
            "still says our workspace has reached the free-plan limit."
        ),
    ),
    _case(
        id="classify-support-fr",
        category="classification",
        title="Classification de ticket support",
        mode="generate",
        language="fr",
        difficulty="easy",
        focus="discipline des labels, confiance, justification courte",
        expected_output="un label autorisé, une confiance et une raison",
        system="Classez avec exactement un label autorisé et n'inventez pas de labels.",
        prompt=(
            "Classez le message avec exactement un des labels suivants : facturation, bug, "
            "fonctionnalité, vente, autre. Retournez `label`, `confidence` et `reason`.\n\n"
            "Message : Nous sommes passés au forfait Pro ce matin, mais le bouton "
            "d'export indique encore que notre espace a atteint la limite du forfait gratuit."
        ),
    ),
    _case(
        id="reasoning-policy-en",
        category="reasoning",
        title="Policy application",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="rule selection, exception handling, concise answer",
        expected_output="decision plus decisive rule",
        system="Reason carefully, cite the decisive rule, and keep the answer short.",
        prompt=(
            "Policy: Refunds are allowed within 30 days unless the item is damaged "
            "by misuse. Enterprise customers may receive store credit up to 60 days "
            "after purchase if a manager approves it.\n\n"
            "Case: An enterprise customer bought hardware 45 days ago. The device "
            "is undamaged, but they ordered the wrong version. What should support offer?"
        ),
    ),
    _case(
        id="reasoning-policy-fr",
        category="reasoning",
        title="Application d'une politique",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="choix de règle, gestion d'exception, réponse concise",
        expected_output="décision et règle décisive",
        system="Raisonnez avec soin, citez la règle décisive et gardez une réponse courte.",
        prompt=(
            "Politique : les remboursements sont autorisés sous 30 jours sauf si "
            "l'article est endommagé par une mauvaise utilisation. Les clients "
            "Enterprise peuvent recevoir un avoir jusqu'à 60 jours après l'achat "
            "si un manager l'approuve.\n\n"
            "Cas : un client Enterprise a acheté du matériel il y a 45 jours. "
            "L'appareil n'est pas endommagé, mais il a commandé la mauvaise version. "
            "Que doit proposer le support ?"
        ),
    ),
    _case(
        id="data-table-insight-en",
        category="data_analysis",
        title="Benchmark table insight",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="numeric comparison, ranking, tradeoff explanation",
        expected_output="winner by use case plus cited numbers",
        system="Compare the numbers carefully and explain the practical tradeoff.",
        prompt=(
            "Analyze this benchmark table. Recommend one model for fast draft answers "
            "and one model for reliable structured output. Cite the numbers that matter.\n\n"
            "| model | avg_latency_s | avg_tokens_s | json_error_rate | summary_score |\n"
            "| qwen3:4b | 2.4 | 41.2 | 0.08 | 8.1 |\n"
            "| gemma3:4b | 3.1 | 34.8 | 0.02 | 8.7 |\n"
            "| llama3.2:3b | 1.7 | 45.5 | 0.16 | 7.4 |"
        ),
    ),
    _case(
        id="data-table-insight-fr",
        category="data_analysis",
        title="Analyse de tableau benchmark",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="comparaison chiffrée, classement, compromis",
        expected_output="gagnant par usage avec chiffres cités",
        system="Comparez les chiffres avec soin et expliquez le compromis pratique.",
        prompt=(
            "Analysez ce tableau de benchmark. Recommandez un modèle pour des brouillons "
            "rapides et un modèle pour une sortie structurée fiable. Citez les chiffres importants.\n\n"
            "| modèle | latence_moy_s | tokens_s_moy | taux_erreur_json | score_résumé |\n"
            "| qwen3:4b | 2.4 | 41.2 | 0.08 | 8.1 |\n"
            "| gemma3:4b | 3.1 | 34.8 | 0.02 | 8.7 |\n"
            "| llama3.2:3b | 1.7 | 45.5 | 0.16 | 7.4 |"
        ),
    ),
    _case(
        id="instruction-format-en",
        category="instruction_following",
        title="Strict output format",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="constraint following, no extra prose, ordering",
        expected_output="exact 4-line response",
        system="Follow formatting instructions exactly.",
        prompt=(
            "Return exactly four lines and nothing else:\n"
            "1. `risk:` followed by the biggest risk in 8 words or fewer\n"
            "2. `owner:` followed by one role\n"
            "3. `deadline:` followed by an ISO date\n"
            "4. `check:` followed by yes or no\n\n"
            "Context: The release is planned for 2026-06-15. The data migration "
            "owner is the platform lead. The rollback checklist is incomplete."
        ),
    ),
    _case(
        id="instruction-format-fr",
        category="instruction_following",
        title="Format de sortie strict",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="respect des contraintes, absence de prose, ordre",
        expected_output="réponse exacte en 4 lignes",
        system="Respectez exactement les consignes de format.",
        prompt=(
            "Retournez exactement quatre lignes et rien d'autre :\n"
            "1. `risk:` suivi du plus grand risque en 8 mots maximum\n"
            "2. `owner:` suivi d'un rôle\n"
            "3. `deadline:` suivi d'une date ISO\n"
            "4. `check:` suivi de yes ou no\n\n"
            "Contexte : la sortie est prévue pour le 2026-06-15. Le responsable "
            "de la migration de données est le lead plateforme. La checklist de "
            "rollback est incomplète."
        ),
    ),
    _case(
        id="translation-tone-en",
        category="translation",
        title="Tone-preserving translation",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="meaning, tone, localization, no additions",
        expected_output="French translation plus one note",
        system="Translate naturally while preserving tone and business meaning.",
        prompt=(
            "Translate this customer email into French for a professional SaaS support team. "
            "Keep the tone calm and helpful. Add one short note if an idiom needs adaptation.\n\n"
            "We hit a snag during the rollout, but your team has been responsive. "
            "Can you confirm whether the audit export will be ready before our board meeting?"
        ),
    ),
    _case(
        id="translation-tone-fr",
        category="translation",
        title="Traduction avec ton préservé",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="sens, ton, localisation, pas d'ajout",
        expected_output="traduction anglaise et une note",
        system="Traduisez naturellement en conservant le ton et le sens métier.",
        prompt=(
            "Traduisez cet email client en anglais pour une équipe support SaaS professionnelle. "
            "Gardez un ton calme et utile. Ajoutez une note courte si une expression doit être adaptée.\n\n"
            "Nous avons rencontré un accroc pendant le déploiement, mais votre équipe a "
            "été réactive. Pouvez-vous confirmer si l'export d'audit sera prêt avant "
            "notre réunion du comité de direction ?"
        ),
    ),
    _case(
        id="code-parser-en",
        category="code_generation",
        title="Python helper with tests",
        mode="generate",
        language="en",
        difficulty="medium",
        focus="correct parsing, edge cases, test coverage",
        expected_output="function plus 5 short tests",
        system="Write correct production-minded Python with tests when asked.",
        prompt=(
            "Write a Python function `parse_size(text: str) -> int` that converts "
            "strings like '5KB', '12 mb', and '1gb' into bytes. Then provide 5 short "
            "test cases."
        ),
    ),
    _case(
        id="code-parser-fr",
        category="code_generation",
        title="Fonction Python avec tests",
        mode="generate",
        language="fr",
        difficulty="medium",
        focus="parsing correct, cas limites, couverture de tests",
        expected_output="fonction et 5 tests courts",
        system="Écrivez du Python fiable et orienté production, avec des tests lorsque demandé.",
        prompt=(
            "Écrivez une fonction Python `parse_size(text: str) -> int` qui convertit "
            "des chaînes comme '5KB', '12 mb' et '1gb' en octets. Fournissez ensuite "
            "5 cas de test courts."
        ),
    ),
    _case(
        id="tool-weather-en",
        category="tool_calling",
        title="Weather tool call",
        mode="chat_tools",
        language="en",
        difficulty="easy",
        focus="tool selection, argument extraction, unit",
        expected_output="get_weather call with city and celsius",
        system="Use the provided tools when they are the best match.",
        prompt="What is the weather in Tokyo right now in celsius?",
        tools=[TOOL_SCHEMAS[0]],
        expected_tool="get_weather",
    ),
    _case(
        id="tool-weather-fr",
        category="tool_calling",
        title="Appel outil météo",
        mode="chat_tools",
        language="fr",
        difficulty="easy",
        focus="sélection d'outil, extraction d'arguments, unité",
        expected_output="appel get_weather avec ville et celsius",
        system="Utilisez les outils fournis lorsqu'ils correspondent le mieux à la demande.",
        prompt="Quel temps fait-il à Tokyo maintenant, en degrés Celsius ?",
        tools=[TOOL_SCHEMAS[0]],
        expected_tool="get_weather",
    ),
    _case(
        id="tool-cart-total-en",
        category="tool_calling",
        title="Cart total tool call",
        mode="chat_tools",
        language="en",
        difficulty="medium",
        focus="tool use, numeric arguments, tax handling",
        expected_output="calculate_total call with item list and tax rate",
        system="Use the provided tools when arithmetic or structured arguments are needed.",
        prompt="Calculate the total for items priced 19.99, 5.50, and 3.25 with a tax rate of 0.0825.",
        tools=[TOOL_SCHEMAS[1]],
        expected_tool="calculate_total",
    ),
    _case(
        id="tool-cart-total-fr",
        category="tool_calling",
        title="Appel outil total panier",
        mode="chat_tools",
        language="fr",
        difficulty="medium",
        focus="usage d'outil, arguments numériques, taxe",
        expected_output="appel calculate_total avec liste d'articles et taux de taxe",
        system="Utilisez les outils fournis lorsque des calculs ou arguments structurés sont nécessaires.",
        prompt="Calculez le total pour des articles à 19.99, 5.50 et 3.25 avec un taux de taxe de 0.0825.",
        tools=[TOOL_SCHEMAS[1]],
        expected_tool="calculate_total",
    ),
    _case(
        id="vision-ui-review-en",
        category="vision",
        title="UI screenshot review",
        mode="vision_generate",
        language="en",
        difficulty="medium",
        focus="visual grounding, issue detection, actionable feedback",
        expected_output="sections plus usability or visual issues",
        system="Describe the image carefully and focus on actionable product feedback.",
        prompt=(
            "Review this uploaded UI image. Identify the main interface sections, "
            "then list usability or visual issues you notice."
        ),
        requires_vision=True,
    ),
    _case(
        id="vision-ui-review-fr",
        category="vision",
        title="Analyse d'une capture UI",
        mode="vision_generate",
        language="fr",
        difficulty="medium",
        focus="ancrage visuel, détection d'issues, retour actionnable",
        expected_output="sections puis problèmes d'utilisabilité ou visuels",
        system="Décrivez l'image avec précision et concentrez-vous sur les retours produit actionnables.",
        prompt=(
            "Analysez cette image d'interface importée. Identifiez les principales "
            "sections de l'interface, puis listez les problèmes d'utilisabilité ou "
            "visuels que vous remarquez."
        ),
        requires_vision=True,
    ),
]


PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "General evaluator": {
        "system": "You are a careful evaluator. Answer directly and clearly.",
        "prompt": "Explain the tradeoffs of running small LLMs locally for a product team.",
    },
    "Comparison summary": {
        "system": "Compare options with evidence and name the best fit for each use case.",
        "prompt": (
            "Compare these model results. Return winners for speed, answer quality, "
            "format reliability, and overall recommendation:\n\n"
        ),
    },
    "Summarization": {
        "system": "Summarize clearly and preserve key facts.",
        "prompt": "Summarize this text in 5 bullets, then give one risk and one next step:\n\n",
    },
    "Structured JSON": {
        "system": "Return only valid JSON. Do not include markdown fences.",
        "prompt": "Extract the following note into JSON with keys `owner`, `deadline`, `priority`, and `action_items`:\n\n",
    },
    "Reasoning": {
        "system": "Reason step by step internally, then return only the conclusion and decisive evidence.",
        "prompt": "Apply the policy to the case. Return `decision`, `rule`, and `short_explanation`:\n\n",
    },
    "Data analysis": {
        "system": "Use the provided numbers. Do not claim unsupported trends.",
        "prompt": "Analyze this table. Identify the strongest option, weakest option, and the main tradeoff:\n\n",
    },
    "Code review": {
        "system": "Act as a senior engineer. Prioritize correctness and risks.",
        "prompt": "Review this code. Return findings first, then a short suggested fix:\n\n",
    },
    "Classification": {
        "system": "Classify the input using only the allowed labels.",
        "prompt": "Classify this customer message as one of: billing, bug, feature, sales, other.\n\nMessage: ",
    },
    "Translation": {
        "system": "Translate naturally and preserve meaning, tone, names, dates, and numbers.",
        "prompt": "Translate into French. Keep product names unchanged and add one short note for any ambiguity:\n\n",
    },
}


def categories() -> list[str]:
    present = {case.category for case in BENCHMARK_CASES}
    ordered = [category for category in CATEGORY_ORDER if category in present]
    return ordered + sorted(present - set(ordered))


def benchmark_languages() -> list[str]:
    present = {case.language for case in BENCHMARK_CASES}
    return [language for language in LANGUAGE_ORDER if language in present]


def cases_by_ids(case_ids: list[str]) -> list[BenchmarkCase]:
    lookup = {case.id: case for case in BENCHMARK_CASES}
    return [lookup[case_id] for case_id in case_ids if case_id in lookup]


def cases_by_categories(
    selected_categories: list[str],
    selected_languages: list[str] | None = None,
) -> list[BenchmarkCase]:
    selected = set(selected_categories)
    languages = set(selected_languages or benchmark_languages())
    return [
        case
        for case in BENCHMARK_CASES
        if case.category in selected and case.language in languages
    ]


def custom_case(
    *,
    title: str,
    mode: str,
    system: str,
    prompt: str,
    tool_schema_text: str = "",
    expected_tool: str = "",
    language: str = "custom",
) -> BenchmarkCase:
    tools = None
    if mode == "chat_tools" and tool_schema_text.strip():
        tools = json.loads(tool_schema_text)
    return BenchmarkCase(
        id=(title.strip() or "custom").lower().replace(" ", "-"),
        category="custom",
        title=title.strip() or "Custom prompt",
        mode=mode,
        system=system.strip(),
        prompt=prompt.strip(),
        language=language,
        difficulty="custom",
        focus="user-defined",
        expected_output="user-defined",
        tools=copy.deepcopy(tools),
        expected_tool=expected_tool.strip() or None,
    )


def default_tool_schema_text() -> str:
    return json.dumps(TOOL_SCHEMAS, indent=2)
