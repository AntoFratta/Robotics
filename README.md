# Assistente Conversazionale Empatico per Pazienti Anziani

Sistema di dialogo guidato basato su LangGraph e Ollama per la raccolta di dati di diario clinico con risposte empatiche personalizzate.

## Descrizione

Questo progetto implementa un assistente conversazionale progettato per interagire con pazienti anziani durante la compilazione di un diario clinico quotidiano. Il sistema utilizza:

- **Empatia Personalizzata**: Risposte adattate al profilo del paziente (età, genere, condizioni di salute, bisogni comunicativi)
- **Analisi Emotiva**: Rilevamento automatico basato su modello di Ekman (Rabbia, Paura, Tristezza, Felicità, Sorpresa, Disgusto, Neutralità)
- **Branching Intelligente**: Follow-up automatici per risposte evasive o emozioni forti con dialogo libero adattivo (max 2 iterazioni)
- **RAG con ChromaDB**: Recupero contestuale di informazioni dal profilo paziente tramite embeddings
- **Logging Strutturato**: Salvataggio sessioni in CSV/JSON per analisi successive

### Architettura

Il sistema usa **LangGraph** per la gestione dello stato dialogico con nodi specializzati:
- Recupero profilo (RAG)
- Input utente
- Classificazione risposta
- Estrazione segnali emotivi
- Generazione empatia personalizzata
- Routing condizionale (follow-up vs prossima domanda)

## Requisiti

### Software Richiesto

1. **Python 3.11+**
   - Scarica da: https://www.python.org/downloads/

2. **Ollama**
   - Piattaforma per eseguire LLM in locale
   - Scarica da: https://ollama.com/download
   - Installazione Windows: esegui `OllamaSetup.exe`
   - Verifica installazione: `ollama --version`

3. **Modelli Ollama Richiesti**
   ```bash
   ollama pull qwen2.5:7b
   ollama pull qwen2.5:3b
   ollama pull mxbai-embed-large
   ```
   - `qwen2.5:7b`: Generazione empatia (4.7 GB)
   - `qwen2.5:3b`: Estrazione segnali emotivi (1.9 GB)
   - `mxbai-embed-large`: Embeddings per RAG (669 MB)

### Dipendenze Python

Le dipendenze sono specificate in `requirements.txt`:
- `langchain-ollama`: Integrazione Ollama
- `langchain-chroma`: Vector store per RAG
- `chromadb`: Database vettoriale
- `langchain-core`: Componenti base LangChain
- `langgraph`: Framework per state machine
- `typing_extensions`: Supporto TypedDict

## Installazione

### 1. Clona il Repository
```bash
git clone https://github.com/AntoFratta/Robotics.git
cd Robotics
```

### 2. Crea Ambiente Virtuale (opzionale ma raccomandato)
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

### 3. Installa Dipendenze
```bash
pip install -r requirements.txt
```

### 4. Verifica Ollama
Assicurati che Ollama sia in esecuzione e i modelli siano scaricati:
```bash
ollama list
```
Dovresti vedere `qwen2.5:7b`, `qwen2.5:3b`, e `mxbai-embed-large`.

## Struttura Progetto

```
Robotics/
├── README.md
├── requirements.txt
├── data/
│   ├── diary_questions.json        # Domande del diario
│   ├── follow_up_questions.json    # Follow-up per temi critici
│   ├── profile_schema.json         # Schema profilo paziente
│   ├── profile_template.json       # Template vuoto
│   └── profiles/                   # Profili pazienti di esempio
│       ├── demo_profile_01.json
│       ├── demo_profile_02.json
│       └── demo_profile_03_minimal.json
├── runtime/
│   ├── chroma_profile_db/          # Database vettoriali (generato)
│   ├── config/                     # Config last selection (generato)
│   └── sessions/                   # Log sessioni (generato)
└── src/
    ├── app.py                      # Entry point
    ├── graph.py                    # LangGraph state machine
    ├── state.py                    # Definizione DialogueState
    ├── profile_store.py            # RAG con ChromaDB
    ├── profile_utils.py            # Gestione profili
    ├── text_utils.py               # Manipolazione testo/genere
    ├── signal_extractor.py         # Estrazione segnali emotivi
    ├── response_classifier.py      # Classificazione risposte
    └── session_logger.py           # Logging CSV/JSON
```

## Uso

### Avvio del Sistema

```bash
python -m src.app
```

### Flusso Interattivo

1. **Selezione Profilo**: Il sistema mostra i profili disponibili in `data/profiles/`
   ```
   Profili disponibili:
   1. Maria Rossi (F, 78 anni) - Parkinson
   2. Giuseppe Verdi (M, 82 anni) - BPCO
   3. Profilo Minimale
   ```

2. **Dialogo Guidato**: Il sistema pone domande del diario e adatta le risposte al profilo
   ```
   DOMANDA 1: Come si è sentito/a oggi?
   Risposta: Bene
   
   ASSISTENTE:
   Mi fa piacere sentirlo.
   ```

3. **Follow-up Automatici**: Se rileva risposte evasive o emozioni forti
   ```
   DOMANDA 5: C'è stato un momento di difficoltà oggi?
   Risposta: Ho paura di cadere quando esco
   
   ASSISTENTE:
   Capisco la sua preoccupazione.
   
   Quando si sente in questo modo, riesce a capire cosa ha scatenato quella sensazione?
   ```

4. **Uscita**: Scrivi `Q` per terminare
   ```
   Sessione salvata:
      CSV: runtime/sessions/P_abc123/2025-12-31_15-30-00.csv
      Metadata: runtime/sessions/P_abc123/2025-12-31_15-30-00_meta.json
      JSON: runtime/sessions/P_abc123/2025-12-31_15-30-00.json
   ```

## Configurazione

### Creare un Nuovo Profilo

1. Copia `data/profile_template.json`
2. Compila i campi seguendo lo schema in `data/profile_schema.json`
3. Salva in `data/profiles/` con nome descrittivo

**Campi Chiave per Personalizzazione**:
- `communication_needs`: "Sente poco, frasi brevi" → risposte più concise
- `living_situation`: Contesto abitativo (vive solo, con familiari, RSA)
- `health_goal`: Obiettivo salute (es: "camminare di più") → collegamento empatia
- `routine_info`: Attività quotidiane → maggiore rilevanza nel RAG

### Modificare Domande del Diario

Modifica `data/diary_questions.json`:
```json
[
  {
    "id": 1,
    "question": "Tua domanda qui",
    "category": "benessere_generale"
  }
]
```

### Personalizzare Follow-up

Modifica `data/follow_up_questions.json` per aggiungere follow-up per emozioni Ekman:
```json
{
  "evasive": ["Domanda per risposte evasive?"],
  "Paura": ["Domanda per paura?"],
  "Rabbia": ["Domanda per rabbia?"],
  "Tristezza": ["Domanda per tristezza?"],
  "Felicità": ["Domanda per felicità?"]
}
```

## Caratteristiche Tecniche

### Gestione Empatia Personalizzata
- **Communication Needs**: Adatta risposte in base a bisogni comunicativi del profilo
- **Emozioni Rilevate**: Tono adattato secondo modello Ekman (7 emozioni base)
- **Contesto Personale**: Usa living_situation e health_goal quando pertinenti
- **Genere**: Accordi grammaticali corretti automatici

### Fast Paths (Template)
- **Risposte 1 parola**: Template fissi per evitare interpretazioni LLM errate
- **Follow-up negativi**: Pattern matching per contesto negativo (es: "vorrei vedere i miei figli")

### Classificazione Risposte
- **Evasive**: "no", "niente", "non ricordo" → dialogo libero con approfondimento
- **Emozioni Forti**: Paura, Rabbia, Tristezza, Felicità → dialogo libero mirato
- **Dialogo Libero**: Massimo 2 iterazioni per domanda, poi passaggio automatico alla successiva
- **Balanced Detection**: Keyword matching + LLM (conflitto risolto con confidence threshold 0.7)

### Estrazione Segnali
- **Approccio Bilanciato**: Keyword matching deterministico + LLM per copertura sinonimi
- **Conflict Resolution**: Se keyword e LLM discordano, usa LLM solo se confidence > 0.7
- **LLM Leggero**: qwen2.5:3b per velocità e stabilità (temperature=0)
- **Output JSON**: emotion (Ekman), intensity, themes, confidence
- **Fallback**: Se parsing fallisce, usa default (Neutralità, confidence 0.3)

## Troubleshooting

### Ollama non risponde
```bash
# Verifica che Ollama sia attivo
ollama list

# Riavvia Ollama (Windows)
Chiudi Ollama dalla system tray e riavvia
```

### Modello non trovato
```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
ollama pull mxbai-embed-large
```

### Errore ChromaDB
Elimina e rigenera il database:
```bash
rmdir /s runtime\chroma_profile_db  # Windows
python -m src.app  # Rigenera automaticamente
```

### Errore import src
Assicurati di eseguire come modulo:
```bash
python -m src.app  # ✓ Corretto
python src/app.py  # ✗ Errore import
```

## Output Sessioni

Ogni sessione genera 3 file in `runtime/sessions/P_<patient_id>/`:

1. **CSV** (`YYYY-MM-DD_HH-MM-SS.csv`): Log conversazione tabellare
2. **Metadata** (`YYYY-MM-DD_HH-MM-SS_meta.json`): Info sessione (durata, profilo)
3. **JSON Completo** (`YYYY-MM-DD_HH-MM-SS.json`): qa_history, branches, signals

## Limitazioni Note

- **Max 2 Iterazioni Dialogo**: Sistema limita dialogo libero a 2 iterazioni per domanda
- **Genere Binario**: Supporta solo M/F per accordi grammaticali
- **Lingua**: Italiano (prompts hardcoded)
- **LLM Locale**: Richiede Ollama in esecuzione (non cloud)
- **Modello Ekman**: Supporta solo 7 emozioni base (no emozioni complesse o miste)