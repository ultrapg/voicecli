import os
import json
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Cache loaded models in-memory
cached_models = {}

def load_config(settings_path=None):
    """
    Loads configuration from settings_path or default settings.json in CWD/next to exe.
    Merges with sensible defaults.
    """
    defaults = {
        "model_name_style": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "model_name_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "device": "cpu",
        "dtype": "float32",
        "huggingface_cache_dir": None,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 2048,
        "repetition_penalty": 1.1
    }
    
    config = defaults.copy()
    resolved_path = None
    
    # 1. Check if explicit settings path provided and exists
    if settings_path and os.path.exists(settings_path):
        resolved_path = settings_path
    else:
        # 2. Check settings.json in CWD
        if os.path.exists("settings.json"):
            resolved_path = "settings.json"
        else:
            # 3. Check settings.json next to current exe
            try:
                import sys
                exe_dir = os.path.dirname(sys.executable)
                exe_settings = os.path.join(exe_dir, "settings.json")
                if os.path.exists(exe_settings):
                    resolved_path = exe_settings
            except Exception:
                pass
                
    if resolved_path:
        print(f"[*] Loading configuration from: {resolved_path}", flush=True)
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                for k, v in user_config.items():
                    if v is not None:
                        config[k] = v
        except Exception as e:
            print(f"[-] Warning: Failed to load config file: {e}. Using defaults.", flush=True)
            
    return config

def get_model(model_name: str, config: dict):
    """
    Loads and caches the specified Qwen3-TTS model in-memory.
    Uses GPU/CUDA/Vulkan/DirectML/CPU as configured.
    """
    global cached_models
    if model_name not in cached_models:
        print(f"[*] Loading model: {model_name} ...", flush=True)
        
        # Configure HuggingFace Cache Dir if provided in settings
        if config.get("huggingface_cache_dir"):
            os.environ["HF_HOME"] = config["huggingface_cache_dir"]
            print(f"[*] Hugging Face cache directory overridden to: {config['huggingface_cache_dir']}", flush=True)
            
        # Parse device
        device_name = config.get("device", "cpu").lower()
        
        # Parse dtype
        dtype_name = config.get("dtype", "float32").lower()
        if dtype_name == "float16":
            dtype = torch.float16
        elif dtype_name == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            
        print(f"[*] Configured device: {device_name} ({dtype})", flush=True)
        
        # We load differently based on backend:
        # CUDA/CPU: native device_map
        # DirectML/Vulkan: load on CPU first, then transfer using .to() to bypass transformers device_map limits.
        model = None
        if device_name == "cuda" or "cuda:" in device_name:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available.")
                device_map = device_name
                model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=dtype
                )
            except Exception as e:
                print(f"[-] Error: Failed to load model on CUDA: {e}. Falling back to CPU.", flush=True)
        elif device_name == "dml":
            try:
                import torch_directml
                dml_device = torch_directml.device()
                print("[*] DirectML backend resolved. Loading model onto DirectML device...", flush=True)
                
                model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map=None,
                    torch_dtype=dtype
                )
                try:
                    model.model.to(dml_device)
                    model.device = dml_device
                except Exception as e:
                    print(f"[-] Error: Failed to transfer model to DirectML device: {e}. Falling back to CPU.", flush=True)
                    model.model.to("cpu")
                    model.device = torch.device("cpu")
            except ImportError:
                print("[-] Error: torch-directml is not installed. Falling back to CPU.", flush=True)
            except Exception as e:
                print(f"[-] Error initializing DirectML: {e}. Falling back to CPU.", flush=True)
        elif device_name == "vulkan":
            if hasattr(torch, "is_vulkan_available") and not torch.is_vulkan_available():
                print("[-] Warning: PyTorch reports Vulkan is not available. Attempting transfer anyway...", flush=True)
            
            print("[*] Loading model on CPU and transferring to Vulkan device...", flush=True)
            try:
                model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map=None,
                    torch_dtype=dtype
                )
                try:
                    model.model.to("vulkan")
                    model.device = torch.device("vulkan")
                except Exception as e:
                    print(f"[-] Error: Failed to transfer model to Vulkan device: {e}. Falling back to CPU.", flush=True)
                    model.model.to("cpu")
                    model.device = torch.device("cpu")
            except Exception as e:
                print(f"[-] Error loading model for Vulkan: {e}. Falling back to CPU.", flush=True)

        if model is None:
            print("[*] Loading model on CPU...", flush=True)
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=dtype
            )
            
        print(f"[*] Model {model_name} loaded successfully.", flush=True)
        cached_models[model_name] = model
        
    return cached_models[model_name]

def generate_style(text: str, prompt: str, output_path: str, settings_path: str = None):
    """
    Generates speech based on target text and an acoustic style description (Voice Design).
    """
    config = load_config(settings_path)
    model_name = config.get("model_name_style", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    
    print(f"[*] Generating speech for text: \"{text}\"", flush=True)
    print(f"[*] Acoustic style description: \"{prompt}\"", flush=True)
    
    try:
        model = get_model(model_name, config)
        
        print("[*] Running inference...", flush=True)
        gen_kwargs = {
            "do_sample": config.get("do_sample", True),
            "temperature": config.get("temperature", 1.0),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "max_new_tokens": config.get("max_new_tokens", 2048),
            "repetition_penalty": config.get("repetition_penalty", 1.1)
        }
        
        wavs, sr = model.generate_voice_design(
            text=text,
            instruct=prompt,
            **gen_kwargs
        )
        
        print(f"[*] Saving audio to {output_path} (Sample Rate: {sr} Hz)...", flush=True)
        sf.write(output_path, wavs[0], sr)
        print("[+] Audio generation complete!", flush=True)
    except Exception as e:
        print(f"[-] Error in generate_style: {e}", flush=True)
        raise

def generate_clone(text: str, audio_path: str, output_path: str, settings_path: str = None):
    """
    Clones a voice from a short reference audio file.
    """
    config = load_config(settings_path)
    model_name = config.get("model_name_clone", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    
    print(f"[*] Cloning voice from reference audio: {audio_path}", flush=True)
    print(f"[*] Generating speech for text: \"{text}\"", flush=True)
    
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
            
        model = get_model(model_name, config)
        
        print("[*] Creating voice clone prompt...", flush=True)
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=audio_path,
            x_vector_only_mode=True
        )
        
        print("[*] Running inference...", flush=True)
        gen_kwargs = {
            "do_sample": config.get("do_sample", True),
            "temperature": config.get("temperature", 1.0),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "max_new_tokens": config.get("max_new_tokens", 2048),
            "repetition_penalty": config.get("repetition_penalty", 1.1)
        }
        
        wavs, sr = model.generate_voice_clone(
            text=text,
            voice_clone_prompt=voice_clone_prompt,
            **gen_kwargs
        )
        
        print(f"[*] Saving audio to {output_path} (Sample Rate: {sr} Hz)...", flush=True)
        sf.write(output_path, wavs[0], sr)
        print("[+] Audio generation complete!", flush=True)
    except Exception as e:
        print(f"[-] Error in generate_clone: {e}", flush=True)
        raise
