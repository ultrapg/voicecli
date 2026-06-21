import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Default model names for style prompting (Voice Design) and voice cloning (Base), can be overridden via env vars
model_name_style = os.getenv("VOICECLI_MODEL_NAME_STYLE", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
model_name_clone = os.getenv("VOICECLI_MODEL_NAME_CLONE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# Cache loaded models in-memory
cached_models = {}

def get_model(model_name: str):
    """
    Loads and caches the specified Qwen3-TTS model in-memory.
    Uses GPU/CUDA if available, falling back to CPU.
    """
    global cached_models
    if model_name not in cached_models:
        print(f"[*] Loading model: {model_name} ...", flush=True)
        
        # Check if CUDA is available, else use CPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"[*] Target device: {device} ({dtype})", flush=True)
        
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype
        )
        print(f"[*] Model {model_name} loaded successfully.", flush=True)
        cached_models[model_name] = model
        
    return cached_models[model_name]

def generate_style(text: str, prompt: str, output_path: str):
    """
    Generates speech based on target text and an acoustic style description (Voice Design).
    """
    print(f"[*] Generating speech for text: \"{text}\"", flush=True)
    print(f"[*] Acoustic style description: \"{prompt}\"", flush=True)
    
    try:
        model = get_model(model_name_style)
        
        print("[*] Running inference...", flush=True)
        # generate_voice_design is the Voice Design API in qwen-tts
        wavs, sr = model.generate_voice_design(
            text=text,
            instruct=prompt
        )
        
        print(f"[*] Saving audio to {output_path} (Sample Rate: {sr} Hz)...", flush=True)
        # sf.write expects (file, data, samplerate)
        # wavs[0] is the first generated audio clip
        sf.write(output_path, wavs[0], sr)
        print("[+] Audio generation complete!", flush=True)
    except Exception as e:
        print(f"[-] Error in generate_style: {e}", flush=True)
        raise

def generate_clone(text: str, audio_path: str, output_path: str):
    """
    Clones a voice from a short reference audio file.
    """
    print(f"[*] Cloning voice from reference audio: {audio_path}", flush=True)
    print(f"[*] Generating speech for text: \"{text}\"", flush=True)
    
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
            
        model = get_model(model_name_clone)
        
        print("[*] Creating voice clone prompt...", flush=True)
        # create_voice_clone_prompt is the cloning API in qwen-tts
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=audio_path,
            x_vector_only_mode=True
        )
        
        print("[*] Running inference...", flush=True)
        wavs, sr = model.generate_voice_clone(
            text=text,
            voice_clone_prompt=voice_clone_prompt
        )
        
        print(f"[*] Saving audio to {output_path} (Sample Rate: {sr} Hz)...", flush=True)
        sf.write(output_path, wavs[0], sr)
        print("[+] Audio generation complete!", flush=True)
    except Exception as e:
        print(f"[-] Error in generate_clone: {e}", flush=True)
        raise
