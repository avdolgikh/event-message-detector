import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import websockets
import json
import os
import asyncio
from typing import List, Tuple, AsyncGenerator

import logging
logging.basicConfig(level=logging.INFO, force=True)


MODEL_DIRECTORY = "oleksiydolgykh/event-message-detector"
RESULTS_DIR = "results"


BUFFER_SIZE = 15  # Number of recent messages to consider
THRESHOLD = 0.01  # Minimum score to consider a message as part of an event
MAX_LOW_SCORES = 3  # Maximum consecutive low-score messages before flushing
MIN_MESSAGES = 3  # Minimum number of messages required for an event to be saved


class EventProcessor:
    """
    Processes incoming messages, identifies high-scoring events, and saves them to files.

    Args:
        model (torch.nn.Module): The trained token classification model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        buffer_size (int): Number of recent messages to consider in the buffer.
        threshold (float): Minimum score for a message to be considered part of an event.
        max_low_scores (int): Maximum low-score messages to include before flushing the event.
        min_messages (int): Minimum number of messages required for an event to be saved.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        buffer_size: int = 15,
        threshold: float = 0.01,
        max_low_scores: int = 3,
        min_messages: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.max_low_scores = max_low_scores
        self.min_messages = min_messages
        self.message_buffer = []  # Stores the latest `buffer_size` messages
        self.event_storage = []  # Stores the current event-related messages
        self.result_counter = 1
        self.low_score_counter = 0

    def add_to_buffer(self, message: dict) -> None:
        """Adds a message to the buffer, maintaining the buffer size."""
        self.message_buffer.append(message)
        if len(self.message_buffer) > self.buffer_size:
            self.message_buffer.pop(0)

    def construct_model_input(self) -> Tuple[str, int, int]:
        """Constructs the input for the model from the buffer."""
        messages = [
            f"[MESSAGE] [{msg['user']}]: {msg['message']}"
            for msg in self.message_buffer
        ]
        concat_messages = "\n".join(messages)
        last_message_end = len(concat_messages)
        last_message_start = last_message_end - len(messages[-1])
        return concat_messages, last_message_start, last_message_end

    def extract_last_message_logits(
        self,
        logits: torch.Tensor,
        offsets: List[Tuple[int, int]],
        last_message_start: int,
        last_message_end: int,
    ) -> torch.Tensor:
        """Extracts logits for the latest message."""
        return logits[[
            idx for idx, (start, end) in enumerate(offsets)
            if (
                (start, end) != (0, 0)
                and last_message_start <= start
                and end <= last_message_end
            )
        ]]

    def score_text(self, input_text: str, last_message_start: int, last_message_end: int) -> float:
        """Scores the input text for the last message."""
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping").squeeze().tolist()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze()[
                inputs["attention_mask"].squeeze().bool()
            ]

        logits = self.extract_last_message_logits(logits, offsets, last_message_start, last_message_end)
        scores = torch.softmax(logits, dim=-1)[:, 1]
        return scores.mean().item()

    def process_message(self, message: dict) -> None:
        """Processes a single message."""
        self.add_to_buffer(message)

        if len(self.message_buffer) < self.buffer_size:
            return  # Wait until buffer is full

        input_text, last_message_start, last_message_end = self.construct_model_input()
        score = self.score_text(input_text, last_message_start, last_message_end)
        logging.info(f"Score: {score}")

        if score > self.threshold:
            self.event_storage.append(message)
            self.low_score_counter = 0
        elif self.event_storage:
            self.low_score_counter += 1
            if self.low_score_counter < self.max_low_scores:
                self.event_storage.append(message)
            else:
                if self.is_event_ready_to_save():
                    result_file = self.save_event_to_file()
                    logging.info(f"Event saved to: {result_file}")
                self.event_storage = []
                self.low_score_counter = 0
    
    def is_event_ready_to_save(self) -> bool:
        """Determines if the current event is ready to be saved."""
        events_to_save = self.get_event_storage_to_save()
        return len(events_to_save) >= self.min_messages

    def get_event_storage_to_save(self) -> list:
        """Returns the current event storage adjusted for low-score messages."""
        return (
            self.event_storage[: -self.low_score_counter]
            if self.low_score_counter > 0
            else self.event_storage
        )

    def save_event_to_file(self) -> str:
        """Saves the current event storage to a file."""
        events_to_save = self.get_event_storage_to_save()
        result_file = os.path.join(RESULTS_DIR, f"event_{self.result_counter:04d}.json")
        with open(result_file, "w") as f:
            json.dump({"lines": events_to_save}, f, indent=4)
        self.result_counter += 1
        return result_file


async def listen(url: str) -> AsyncGenerator[dict, None]:
    """Listens to the WebSocket stream and yields messages."""
    while True:
        try:
            async with websockets.connect(url, ping_timeout=60, close_timeout=10) as websocket:
                async for message in websocket:
                    yield json.loads(message)
        except (websockets.exceptions.ConnectionClosedError, Exception) as e:
            logging.error(f"Error: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)


async def run(processor: EventProcessor, message_generator: AsyncGenerator[dict, None]) -> None:
    """Processes messages from the generator using the processor."""
    async for message in message_generator:
        processor.process_message(message)


def load_model() -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIRECTORY)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY)
    tokenizer.truncation_side = "left"
    return model, tokenizer


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Event message detector.")
        parser.add_argument("--url", required=True, help="WebSocket URL")
        args = parser.parse_args()
        stream_url = args.url

        model, tokenizer = load_model()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        processor = EventProcessor(
            model,
            tokenizer,
            BUFFER_SIZE,
            THRESHOLD,
            MAX_LOW_SCORES,
            MIN_MESSAGES
        )
        message_generator = listen(stream_url)
        asyncio.run(
            run(processor, message_generator)
        )

    except Exception as e:
        logging.exception(f"Error in main: {e}")
