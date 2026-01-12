# Edge-RAG Pipeline - Setup Guide

**Masterthesis**: "Enhancing Reasoning Fidelity in Quantized Small Language Models on Edge Devices"

---

## 📋 Voraussetzungen

- **Python**: 3.10+
- **Ollama**: Lokal installiert (für SLM + Embeddings)
- **RAM**: Mindestens 8GB (für Phi-3 + nomic-embed-text)
- **OS**: Linux, macOS, Windows (mit WSL empfohlen)

---

## 🚀 Installation (5 Minuten)

### 1. Virtuelle Umgebung erstellen

```bash
python3.10 -m venv edge_rag_env
source edge_rag_env/bin/activate  # Linux/macOS
# oder
edge_rag_env\Scripts\activate  # Windows
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Ollama Modelle herunterladen

```bash
# In separatem Terminal
ollama serve

# In neuem Terminal
ollama pull phi3          # ~2.3 GB
ollama pull nomic-embed-text  # ~275 MB
```

Verifizierung:
```bash
ollama list
```

Erwartet:
```
phi3:latest           2.3 GB
nomic-embed-text:latest  275 MB
```

---

## 📁 Projektstruktur

```
edge-rag-project/
├── config/
│   └── settings.yaml         # Zentrale Konfiguration
├── src/
│   ├── __init__.py
│   ├── ingestion.py          # PDF → Chunks
│   ├── storage.py            # Vector DB + Graph
│   └── retrieval.py          # Hybrid Retrieval
├── data/
│   ├── documents/            # PDF-Dateien HIER einfügen
│   ├── vector_db/            # LanceDB (auto-created)
│   └── knowledge_graph/      # NetworkX Graph (auto-created)
├── logs/
│   └── edge_rag.log         # Logs
├── main.py                  # Entry Point
├── requirements.txt
└── SETUP.md
```

---

## 🎯 Quick Start

### 1. PDFs vorbereiten

```bash
mkdir -p data/documents
cp your_thesis.pdf data/documents/
cp another_paper.pdf data/documents/
```

**Unterstützte Formate**: PDF

### 2. Pipeline ausführen

```bash
python main.py
```

**Output**:
```
2024-01-10 10:15:23 - root - INFO - Edge-RAG Pipeline Start
2024-01-10 10:15:24 - src.ingestion - INFO - DocumentIngestionPipeline initialisiert: chunk_size=512, overlap=128
2024-01-10 10:15:25 - src.ingestion - INFO - Lade PDF: thesis.pdf
2024-01-10 10:15:27 - src.storage - INFO - LanceDB initialisiert: ./data/vector_db
2024-01-10 10:15:28 - src.retrieval - INFO - HybridRetriever initialisiert: mode=RetrievalMode.hybrid
2024-01-10 10:15:29 - root - INFO - Retrieval Results...
```

### 3. Logs inspizieren

```bash
tail -f logs/edge_rag.log
```

---

## ⚙️ Konfiguration anpassen

**Datei**: `config/settings.yaml`

### SLM-Modell wechseln

```yaml
llm:
  model_name: "mistral"  # statt phi3
  # oder "orca", "neural-chat", etc.
```

Verfügbare Modelle (via Ollama):
- `phi3` (2.3 GB) - Schnell, sparsam
- `mistral` (4.2 GB) - Bessere Qualität
- `orca` (7.0 GB) - State-of-the-Art

### Chunk-Größe optimieren

```yaml
chunking:
  chunk_size: 256      # Kleiner = mehr Chunks
  chunk_overlap: 64    # Weniger Overlap = weniger Redundanz
```

**Hinweis für Thesis**: 
- Größere Chunks: Bessere Kontextkohärenz, weniger Chunks
- Kleinere Chunks: Schnellere Retrieval, mehr Noise

### Retrieval-Modus wechseln

```yaml
rag:
  retrieval_mode: "vector"  # "vector", "graph", oder "hybrid"
  vector_weight: 1.0        # Nur Vector
  graph_weight: 0.0         # Keine Graph
```

---

## 🧪 Debugging

### Ollama-Fehler

```bash
# Prüfe ob Ollama läuft
curl http://localhost:11434/api/tags

# Falls nicht:
ollama serve
```

### Keine PDFs gefunden

```bash
ls -la data/documents/
```

### Memory-Fehler

```yaml
# settings.yaml
performance:
  batch_size: 2    # Reduzieren
  device: "cpu"    # (GPU wenn verfügbar)
```

---

## 📊 Wissenschaftliche Validierung

### Ablation Studies durchführen

```python
# In main.py, modifiziere:
retrieval_config = RetrievalConfig(
    mode=RetrievalMode.VECTOR,      # Nur Vektor
    vector_weight=1.0,
    graph_weight=0.0,
)
# → Performance Benchmark durchführen
```

### Metrics sammeln

```bash
# Logs enthalten:
# - Retrieval Latency
# - Vector/Graph Score Distribution
# - Chunk Coverage

grep "Retrieval" logs/edge_rag.log
```

---

## 🔗 Integration in VS Code

### 1. Python Extension
- Install: Microsoft Python Extension

### 2. Launch Config (.vscode/launch.json)

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Edge-RAG Main",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

### 3. VS Code Shortcut
- F5: Run
- Shift+F5: Stop

---

## 📝 Nächste Schritte für Thesis

1. **Data Ingestion Phase**
   - PDFs in `data/documents/` platzieren
   - `main.py` ausführen
   - Vector DB + Graph popeln

2. **Retrieval Evaluation**
   - Queries testen
   - Precision/Recall messen
   - Ablation Studies durchführen

3. **RAG Generation**
   - Ollama LLM Integration (separate `generation.py`)
   - Prompt Engineering
   - Quality Evaluation (BLEU, ROUGE, Manual)

4. **Quantization Analysis**
   - 4-bit vs 8-bit Benchmarks
   - Latency Profiling
   - Token-Accuracy Tradeoff

---

## 🆘 Troubleshooting

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError: No module named 'lancedb'` | `pip install -r requirements.txt` |
| `ConnectionError: http://localhost:11434` | `ollama serve` in neuem Terminal |
| `FileNotFoundError: ./config/settings.yaml` | `mkdir config` dann YAML kopieren |
| `Out of memory` | `batch_size` in settings.yaml reduzieren |
| PDF nicht geladen | Prüfe: `ls data/documents/*.pdf` |

---

## 📚 Weiterführende Ressourcen

- **RAG Surveys**: Gao et al. 2023 (Retrieval-Augmented Generation Overview)
- **Graph-RAG**: Yu et al. 2024 (Graphs for RAG)
- **Chunking Strategies**: LangChain Documentation
- **Ollama**: https://ollama.ai
- **LanceDB**: https://lancedb.com

---

**Viel Erfolg bei der Thesis!**

Bei Fragen: Siehe `logs/edge_rag.log` für detaillierte Debug-Ausgaben.