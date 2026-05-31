# NuroAI: Stateful Symptom Scoring & Diagnostic Inference Engine
NuroAI is an experimental Python-based diagnostic inference engine designed to map unstructured patient symptom descriptions into ranked medical condition predictions. Built entirely without external LLM APIs, the system combines fuzzy symptom matching, weighted condition scoring, conversational state management, and follow-up question generation to explore the capabilities and limitations of deterministic medical triage systems.

Rather than functioning as a clinical diagnostic tool, NuroAI was developed as an engineering exercise in natural language processing, stateful decision-making, and rule-based inference under constrained computational environments.

**NuroAI was ultimately less valuable as a diagnostic system than as an exploration of the limitations of deterministic language-processing pipelines, an insight that directly motivated later work in embedded DSP and TinyML.**

## Key Features
- 59 supported medical conditions
- 246 weighted symptom-condition mappings
- 129 unique symptom indicators
- Stateful multi-turn conversation management
- Severity-aware symptom weighting
- Dynamic follow-up question generation
- Urgency classification framework
- Fully local inference with no cloud dependencies


## Known Limitations

### 1. Vocabulary Variability
Users frequently describe identical symptoms using dramatically different language.
Examples:
- "Head feels like it's being squeezed"
- "Pressure behind my eyes"
- "My skull is pounding"
Although these descriptions may refer to the same underlying symptom, they require extensive synonym coverage and fuzzy matching logic for consistent interpretation.

### 2. Symptom Overlap
Many medical conditions share common symptoms such as:
- fatigue
- fever
- nausea
- dizziness
This often results in closely clustered condition scores and increases the need for follow-up questions to improve discrimination.

### 3. Sparse Inputs
Minimal symptom descriptions such as:
"I feel tired."
provide insufficient information for reliable condition differentiation and often produce broad rankings across multiple conditions.

### 4. Rule-Based Scaling Constraints
As the number of supported conditions grows, maintaining symptom dictionaries, weighting schemes, and follow-up logic becomes increasingly difficult. This creates significant scalability challenges for purely rule-based systems.

### 5. Linguistic Ambiguity
Human language is inherently variable and difficult to compress into deterministic symptom representations. As conversational complexity increases, maintaining robust interpretation without large-scale language models becomes increasingly challenging.

**Engineering Pivot**: NuroAI served as a deep exploration into the limitations of deterministic language-processing systems. Building symptom extraction pipelines, fuzzy-matching routines, weighted scoring frameworks, and conversational state management exposed the challenges of mapping unstructured human language onto finite decision structures. This realization ultimately motivated my transition toward embedded intelligence and digital signal processing, where physical systems generate measurable, repeatable signals that can be analyzed using deterministic DSP pipelines and TinyML models running entirely on-device.

## Core Architecture

### 1. Symptom Extraction & Matching
- Normalizes raw patient input.
- Extracts symptom candidates using a curated symptom vocabulary.
- Applies fuzzy matching via difflib to account for spelling errors, phrasing variations, and approximate symptom descriptions.

### 2. Context-Aware Patient Modeling
The system maintains a persistent patient context capable of tracking:
- Identified symptoms
- Symptom severity
- Symptom duration
- Patient age
- Patient gender
- Previous conversation state
This enables continuity across multiple conversational turns.

### 3. Weighted Condition Scoring
- Matches extracted symptoms against condition profiles stored in medical_diseases.json.
- Applies weighted symptom contributions.
- Aggregates condition scores.
- Filters low-confidence matches using predefined thresholds.

### 4. Dynamic Follow-Up Routing
When multiple candidate conditions produce similar scores, NuroAI generates predefined discriminating follow-up questions designed to collect additional evidence and improve ranking accuracy.

### 5. Urgency Assessment & Safety Logic
The engine includes a basic urgency classification framework capable of categorizing cases into:
- Emergency
- High Priority
- Moderate Priority
- Low Priority
based on symptom combinations and severity indicators.

Unsupported, ambiguous, or low-confidence inputs are routed to clarification prompts rather than producing unsupported diagnostic outputs.

## Techology Stack
- Language: Python 3.8+
- Backend Framework: Flask
- Numerical Computing: NumPy
- Machine Learning Utilities: Scikit-Learn
- Fuzzy Matching: Difflib
- Model Serialization: Joblib
- Data Storage: JSON-based symptom and condition databases

## Processing Pipeline
User Input
–->
Symptom Extraction & Fuzzy Matching
–->
Patient Context Update
–->
Severity & Duration Parsing
–->
Weighted Condition Scoring
–->
Threshold Filtering
–->
Condition Ranking
–->
Follow-Up Question Selection
–->
Urgency Assessment
–->
JSON API Output


## Engineering Retrospective

### The Linguistic Bottleneck
Building NuroAI highlighted the difficulty of mapping open-ended human language onto deterministic computational structures. Real patient descriptions contain virtually unlimited vocabulary variations, making comprehensive rule coverage difficult to achieve without substantial language modeling infrastructure.

### From Language to Signals
This realization ultimately shifted my engineering focus toward embedded sensing and digital signal processing. Unlike natural language, physical systems produce deterministic signals governed by measurable physical processes. Mechanical failures, vibration signatures, and acoustic phenomena can be represented as stable frequency-domain patterns, making them significantly more suitable for constrained edge inference.

The scoring architectures, filtering strategies, and decision frameworks developed during NuroAI directly influenced the design philosophy behind Aura Diagnostics, where similar concepts are applied to real-time DSP pipelines and TinyML inference running on ARM Cortex-M microcontrollers entirely offline.

## Educational & Clinical Disclaimer
NuroAI is an experimental educational project developed for studying data structures, stateful inference systems, conversational workflows, and algorithmic decision-making. It has not been clinically validated, does not process real biometric signals, and must never be used as a substitute for professional medical advice, diagnosis, or treatment.

## Configuration & Usage

### Expanding the Knowledge Base
Additional conditions may be added directly to medical_diseases.json. New symptom indicators should include appropriately weighted relevance values to ensure consistent integration into the scoring framework.

### Running the Server
> python3 app.py 
- The Flask API exposes a /predict endpoint capable of receiving structured symptom inputs and returning ranked condition scores, urgency classifications, and follow-up recommendations.
