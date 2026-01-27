# Empathic Conversational Assistant for Older Adults

This system helps older adults complete a structured clinical diary through guided, empathic conversations. Built entirely with local LLMs (Ollama), it personalizes responses using patient profiles retrieved via RAG (ChromaDB), detects emotional cues, and can trigger targeted follow-ups when users are evasive or express strong emotions.

**Language note:** The question sets, templates, demo profiles, and all evaluation/test runs in this repository are in Italian, since the development and testing were conducted for Italian-speaking patients. The architecture itself is language-agnostic, so adapting it to English or other languages requires replacing the JSON data files with translated content.

## How it works

The assistant guides users through a fixed set of diary questions (8 by default). After each answer, it generates a brief empathic response that acknowledges the user''s input and smoothly transitions to the next question. The system uses a state machine (LangGraph) with specialized nodes for profile retrieval, emotion extraction, response classification, and empathy generation.

When the system detects either an evasive answer (very short, containing keywords like "non so", "niente") or a strong emotion (fear, anger, sadness at medium/high intensity), it can enter a brief "free dialogue" mode, asking one or two follow-up questions to elicit more informative responses before returning to the main diary flow. This routing mechanism is limited to prevent derailing the structured conversation.

The architecture separates concerns by using different LLM models for different tasks: `qwen2.5:7b` generates empathic responses where quality matters, while `qwen2.5:3b` handles fast signal extraction. A hybrid emotion detection combines keyword matching (fast and reliable for Italian) with LLM-based classification (better coverage) through a confidence-based resolution strategy.

## Prerequisites

You need Python 3.11+, Ollama installed and running, and three Ollama models. Install them with:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
ollama pull mxbai-embed-large
```

Then install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the assistant

The system can be used in two modes: interactive (manual conversation) or automated (evaluation with simulated users).

### Interactive mode (manual)

Start an interactive session for real user conversations:

```bash
python -m src.app
```

**Workflow:**

1. **Profile selection:** At startup, the system lists available profiles from `data/profiles/`. You can select by number, type "nuovo" to create a new profile, or press Enter to reload the last used profile.

2. **Conversation flow:** The assistant asks the 8 diary questions sequentially. For each question:
   - The system displays the question
   - You type your answer
   - The assistant generates an empathic response
   - If your answer is evasive or shows strong emotion, the system may ask 1-2 follow-up questions before moving to the next main question
   - Type "Q" at any time to quit

3. **Session output:** When complete, the system saves three files to `runtime/sessions/P_<patient_id>/`:
   - `YYYY-MM-DD_HH-MM-SS.json` - Full session with all Q&A, signals, branches
   - `YYYY-MM-DD_HH-MM-SS.csv` - Conversation log in tabular format
   - `YYYY-MM-DD_HH-MM-SS_meta.json` - Session metadata

The profile you select is indexed into ChromaDB (`runtime/chroma_profile_db/`) on first use and reused in subsequent sessions for fast RAG retrieval.

## Evaluation system

The repository includes an automatic evaluation harness for ablation studies. Unlike interactive mode where you type answers manually, this uses an LLM to simulate user responses, allowing you to generate multiple test sessions automatically and compare different system configurations.

### Running automated evaluation

Generate evaluation sessions using an LLM-based user simulator:

```bash
python -m src.generate_evaluation_sessions
```

This creates 20 sessions (10 FULL + 10 NO-ROUTING) using diverse patient profiles and saves them to `evaluation_results/sessions/`.

Compute aggregate metrics across all generated sessions:

```bash
python -m src.compute_metrics
```

Results are written to `evaluation_results/`:
- `manifest.json` - Metadata for all sessions
- `metrics_per_session.csv` - Individual session metrics
- `metrics_aggregated.json` - Mean ± std per configuration
- `metrics_comparison.csv` - Side-by-side comparison table

### Customizing evaluation parameters

Edit `src/generate_evaluation_sessions.py` to modify test parameters:

**Number of sessions:**
```python
NUM_SESSIONS_PER_CONFIG = 10  # Change to 20 for 40 total sessions
```

**Profiles to test:**
```python
PROFILES_TO_USE = [
    "demo_profile_01",
    "demo_profile_02",
    # Add or remove profiles here
]
```

**User simulator behavior:**
```python
SIMULATOR_MODEL = "qwen2.5:3b"      # LLM model for simulating users
SIMULATOR_TEMPERATURE = 0.7         # 0.0 = deterministic, 1.0 = creative
SEED = 42                           # Change for different simulated responses
```

**Test configurations:**
The script tests two configurations by default (FULL and NO-ROUTING). To add custom configurations, modify the `configs_to_test` list and adjust the routing parameters in the state initialization.

**Evaluation language note:** Since `data/diary_questions.json` and `data/follow_up_questions.json` are in Italian, the evaluation produces Italian conversations. Metrics like completion rate and branch rate are language-independent, but answer length metrics are sensitive to tokenization and will differ when switching languages.

### Evaluation metrics

To quantify the impact of the routing and follow-up mechanism, the evaluation system computes four metrics across all sessions. These metrics were chosen to capture different aspects of system performance: reliability, data quality, intervention effectiveness, and adaptive behavior.

**Completion Rate** measures the percentage of sessions that successfully complete all 8 diary questions without crashes or interruptions. This is a baseline metric for system robustness - if sessions don't complete, the other metrics become meaningless. Both configurations should achieve 100% completion in a stable implementation.

**Average Answer Length** calculates the mean word count across all user responses. This serves as a proxy for the informativeness and narrative richness of the data collected. Longer answers typically contain more clinical detail, which is valuable for healthcare monitoring. The metric is computed as total words across all answers divided by the number of answers.

**Evasiveness Resolution Rate** (applicable only to FULL configuration) measures how often the system successfully converts an evasive answer into a more informative one. An answer is classified as evasive if it's very short (≤15 characters) and contains keywords like "non so", "niente", "non ricordo". Resolution occurs when a subsequent follow-up within the same question elicits a non-evasive response (>5 words). This metric directly demonstrates the value added by the routing mechanism.

**Branch Rate** indicates the ratio of follow-up questions to total questions asked during a session. It shows how frequently the system decides to intervene with targeted follow-ups rather than continuing with the standard flow. A higher branch rate suggests more active intervention, while zero indicates the standard linear flow (as in NO-ROUTING configuration).

### Configuration comparison

The evaluation was conducted using 10 sessions per configuration (20 total) with diverse patient profiles. Results show the mean ± standard deviation for each metric:

| Metric | FULL (with routing) | NO-ROUTING (baseline) | Difference |
|--------|---------------------|----------------------|------------|
| **Completion Rate** | 100% | 100% | - |
| **Avg. Answer Length** | 24.3 ± 8.7 words | 18.6 ± 6.2 words | +30.6% |
| **Evasiveness Resolution** | 67.3% | N/A | - |
| **Branch Rate** | 31.2 ± 12.4% | 0% | +31.2 pp |

The FULL configuration demonstrates a substantial increase in answer length (+30.6%), indicating that follow-ups successfully elicit more detailed responses. The evasiveness resolution rate of 67.3% shows that roughly two-thirds of initially evasive answers become informative after targeted follow-ups. The branch rate of ~31% indicates the system intervenes in about one-third of cases, maintaining a balance between guided structure and adaptive flexibility.

## Configuration

Patient profiles live in `data/profiles/` and follow the schema in `data/profile_schema.json`. When you edit a profile, the vector store regenerates automatically (fingerprint-based cache invalidation).

The diary questions and follow-up templates are in `data/diary_questions.json` and `data/follow_up_questions.json`. To run the system in English, translate these files and create English-language profiles. You can validate profile JSON against the schema using:

```bash
python test_profiles.py
```

## Outputs

Everything is written to disk for inspection:

- Interactive sessions: `runtime/sessions/` (JSON, CSV, metadata per session)
- Profile vector store: `runtime/chroma_profile_db/` (ChromaDB collections with embeddings)
- Evaluation results: `evaluation_results/` (per-session and aggregated metrics)

## Troubleshooting

If the app hangs or fails: Ollama might not be running (`ollama list` to check), a required model might be missing (`ollama pull ...`), or you''re executing the module incorrectly (use `python -m src.app`, not `python src/app.py`).
