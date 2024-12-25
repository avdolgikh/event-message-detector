# Event Message Detector

## Overview
This project processes real-time message streams to detect event-related conversations, such as scheduling meetings. Detected events are saved as JSON files. The application leverages Hugging Face transformers, a fine-tuned token classification model, and WebSocket for data streaming.

## Input Message Format
Messages have the following format:

```json
{
  "seqid": 332570,
  "ts": 1704100456.8682837,
  "user": "darkenigma2652",
  "message": "ign0ramus: I'll start off with KDE so they can get used to Linux, and if it's too slow I'll install XFCE and tweak it a bit"
}
```

### Fields
| Field    | Description                                     |
|----------|-------------------------------------------------|
| `seqid`  | Unique identifier for the message               |
| `ts`     | Timestamp of when the message was sent          |
| `user`   | Username of the sender                          |
| `message`| Content of the message                          |

## Task
The system detects conversations that can be converted into calendar events. Such events may span multiple messages. Detected events are stored as sequentially numbered JSON files under the `results` directory.

Example event-related conversation:

| seqid   | ts             | user     | message                                      |
|---------|----------------|----------|----------------------------------------------|
| 45625   | 1704100456.66  | e_t_     | hstefan: yeah? want to jump on a call?       |
| 45626   | 1704100456.76  | flavio   | I could join if you’re doing screen share.   |
| 45627   | 1704100456.86  | hstefan  | that sounds great - google meet or discord?  |
| 45628   | 1704100456.96  | e_t_     | discord works for me. how about this evening?|
| 45629   | 1704100457.06  | flavio   | +1 for discord, 8pm UTC?                     |

Saved JSON example:

```json
{
  "lines": [
    {
      "seqid": 45625,
      "ts": 1704100456.66,
      "user": "e_t_",
      "message": "hstefan: yeah? want to jump on a call?"
    },
    {
      "seqid": 45626,
      "ts": 1704100456.76,
      "user": "flavio",
      "message": "I could join if you’re doing screen share."
    }
  ]
}
```

## Prerequisites
- Python 3.11
- Poetry for dependency management
- Docker (optional for containerization)

## Installation
### Clone the Repository
```bash
git clone https://github.com/avdolgikh/event-message-detector.git
cd event-message-detector
```

### Install Dependencies
Using Poetry:
```bash
poetry install
```

### Model
- The Hugging Face model is downloaded automatically during runtime from Hugging Face Hub.
- https://huggingface.co/oleksiydolgykh/event-message-detector

## Running the Application
### Command-Line Usage
Run the inference script with a WebSocket URL:
```bash
poetry run python app/inference.py --url ws://143.110.238.245:8000/stream
```

### Docker Usage
#### Build the Docker Image
```bash
docker build -t event-message-detector .
```

#### Run the Docker Container
With volume mounting for results:
```bash
docker run --rm -it -v $(pwd)/results:/app/results event-message-detector --url ws://143.110.238.245:8000/stream
```
Without volume mounting:
```bash
docker run --rm -it event-message-detector --url ws://143.110.238.245:8000/stream
```

## Development Workflow
### Data Collection
- Connect to the WebSocket stream and collect historical data.
- Save messages in Parquet format for analysis.

### Data Labeling
Use GPT to label event-related messages. Messages are processed in overlapping chunks to ensure continuity.

### Model Training
1. Split data into `train`, `valid`, and `test` sets.
2. Fine-tune a Hugging Face transformer model with token classification.
3. Evaluate using metrics like ROC-AUC and F1-score.
4. Save the trained model to the Hugging Face Hub.

### Inference Pipeline
1. Process incoming messages in real time.
2. Buffer recent messages to construct model input.
3. Predict event scores and save high-confidence events to JSON files.

## Directory Structure
```
.
├── app/
│   ├── inference.py            # Inference script
│   └── ...
├── results/                    # Directory for event-related JSON files
├── training/
│   ├── data_labeling.ipynb     # Data gathering and labeling
│   └── model_training.ipynb    # Model training 
├── pyproject.toml              # Poetry configuration
├── Dockerfile                  # Docker container setup
└── README.md                   # Project documentation
```

## Model Details
- **Base Model**: `xlm-roberta-base` (multilingual, 275M params)
- **Task**: Token classification for event detection
- **Tokenizer**: Truncates from the left to prioritize recent messages
- **Training**:
  - Sliding window of 15 messages
  - Weighted cross-entropy loss

## Metrics
- **Validation**:
  - During training, using AUC-ROC for the last message scores. **AUC-ROC = 0.97**. 
  - For finding the best binarization threshold, studying F1, Precision, Recall, selecting the threshold for high Recall.
- **Test Set**: Calculating F1, Precision, Recall at the optimal threshold.
  - For **threshold = 0.01**, Precision = 0.59, while **Recall = 0.84**.


## Future Work
- Weaken the assumption that event conversations are dense and almost no irrelevant messages in between.
- Pay more attention to the fact of multilingual messages
- In case of real chatting, leverage delay between messages as useful feature 
- Try other approaches (token classification/text segmentation is only one), base models, etc.
- Try intensive data cleaning and preprocessing
- Construct a thorough F-beta metric - to define the business vision on Precision vs Recall trade-off
- Durirng detection, consider detect a few recent message in context with very last one (rescore the ones)
- DSpy for data labeling prompts
- Pydantic for data labeling response control
- Cover with unit tests

---

Happy detecting!
