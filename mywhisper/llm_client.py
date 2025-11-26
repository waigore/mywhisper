"""
LLM client utilities for mywhisper.
"""

from __future__ import annotations

import abc
import logging

import requests

LOGGER = logging.getLogger("mywhisper.llm_client")


class LLMClient(abc.ABC):
    """
    Abstract base class for LLM clients.
    """

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        
        Args:
            prompt: The input prompt string
            
        Returns:
            The generated response string
        """
        pass


class OllamaClient(LLMClient):
    """
    Client for interacting with Ollama API.
    """

    def __init__(self, model: str, endpoint: str = "http://localhost:11434/api/generate") -> None:
        """
        Initialize Ollama client.
        
        Args:
            model: The model name to use (e.g., "llama3")
            endpoint: The Ollama API endpoint URL
        """
        self.model = model
        self.endpoint = endpoint
        self.logger = LOGGER.getChild("OllamaClient")

    def generate(self, prompt: str) -> str:
        """
        Generate a response using Ollama API.
        
        Args:
            prompt: The input prompt string
            
        Returns:
            The generated response string
        """
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.RequestException as e:
            self.logger.error("Ollama API request failed: %s", e)
            raise

