from typing import Optional, Dict, List, Generator
import logging
import subprocess
import json

logger = logging.getLogger(__name__)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not installed, will use subprocess fallback")


class LLMClient:
    def __init__(
        self,
        model: str = "llama3.2:latest",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        self._verify_model()
        logger.info(f"LLMClient initialized with model={model}")
    
    def _verify_model(self):
        try:
            if OLLAMA_AVAILABLE:
                models = ollama.list()
                model_names = [m['name'] for m in models.get('models', [])]
                model_base = self.model.split(':')[0]
                if not any(model_base in name for name in model_names):
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                    logger.info(f"Attempting to pull model {self.model}...")
                    self._pull_model()
            else:
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True
                )
                if self.model.split(':')[0] not in result.stdout:
                    logger.warning(f"Model {self.model} may not be available")
                    self._pull_model()
        except Exception as e:
            logger.warning(f"Could not verify model: {e}")
    
    def _pull_model(self):
        try:
            logger.info(f"Pulling model {self.model}...")
            if OLLAMA_AVAILABLE:
                ollama.pull(self.model)
            else:
                subprocess.run(['ollama', 'pull', self.model], check=True)
            logger.info(f"Model {self.model} pulled successfully")
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            if OLLAMA_AVAILABLE:
                return self._generate_ollama(
                    prompt, system_prompt, max_tokens, temperature, stream
                )
            else:
                return self._generate_subprocess(
                    prompt, system_prompt, max_tokens, temperature
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str:
        messages = []
        
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        messages.append({'role': 'user', 'content': prompt})
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': temperature,
                'num_predict': max_tokens
            },
            stream=stream
        )
        
        if stream:
            full_response = ""
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
            return full_response
        else:
            return response['message']['content']
    
    def _generate_subprocess(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        result = subprocess.run(
            ['ollama', 'run', self.model, full_prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            raise Exception(f"Ollama failed: {result.stderr}")
        
        return result.stdout.strip()
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        if not OLLAMA_AVAILABLE:
            yield self.generate(prompt, system_prompt)
            return
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        if OLLAMA_AVAILABLE:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            return response['message']['content']
        else:
            prompt = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in messages
            ])
            return self._generate_subprocess(prompt, None, max_tokens, temperature)
    
    def summarize(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        prompt = f"""Summarize the following text in about {max_length} words or less. 
Focus on the main points and key information.

Text:
{text}

Summary:"""
        
        return self.generate(prompt, max_tokens=max_length * 2)
    
    def set_model(self, model: str):
        self.model = model
        self._verify_model()
        logger.info(f"Model changed to {model}")
    
    def set_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
    
    def health_check(self) -> bool:
        try:
            response = self.generate("Say 'OK' if you're working.", max_tokens=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

