import os
import sys
import json
import time
import mimetypes
from pathlib import Path

# Fix for Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can import requests
sys.path.append(r"c:\Users\35401\project\ai-Training\LTX-2\.venv\Lib\site-packages")

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it: uv pip install requests")
    sys.exit(1)

def get_api_key():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and len(sys.argv) > 2:
        api_key = sys.argv[2]
    
    if not api_key:
        print("Error: API Key not found.")
        sys.exit(1)
    return api_key

def robust_request(method, url, **kwargs):
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            
            if response.status_code == 429:
                print(f"\n  [Rate Limit] Hit 429. Waiting 60 seconds...", end="\r")
                time.sleep(60)
                continue # Retry immediately after sleep without counting against max_retries? Or just count it?
                # Better to count it but give many retries.
            
            # Check for fatal 400 errors (API Key issues)
            if response.status_code == 400:
                try:
                    err_json = response.json()
                    msg = err_json.get("error", {}).get("message", "")
                    reason = ""
                    if "details" in err_json.get("error", {}):
                         reason = err_json.get("error", {}).get("details", [{}])[0].get("reason", "")
                    
                    if "API key" in msg or "API_KEY_INVALID" in reason:
                        print(f"\n\nFATAL ERROR: {msg}")
                        print("Please provide a valid, active API Key.")
                        sys.exit(1)
                except:
                    pass

            # If 5xx error, retry.
            if 500 <= response.status_code < 600:
                 response.raise_for_status() 
            return response
        except (requests.exceptions.RequestException, Exception) as e:
            if i == max_retries - 1:
                raise e
            
            wait_time = 5 * (i + 1)
            print(f"  [Retry {i+1}/{max_retries}] Network/Server error: {e}. Waiting {wait_time}s...", end="\r")
            time.sleep(wait_time)
    return None

def upload_file(api_key, file_path):
    file_size = file_path.stat().st_size
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "video/mp4"
    
    # 1. Start Upload
    upload_url_endpoint = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json"
    }
    data = {"file": {"display_name": file_path.name}}
    
    try:
        response = robust_request("POST", upload_url_endpoint, headers=headers, json=data)
        response.raise_for_status()
        upload_url = response.headers.get("X-Goog-Upload-URL")
        
        if not upload_url:
            raise Exception("No X-Goog-Upload-URL in response")
            
        # 2. Upload Bytes
        with open(file_path, "rb") as f:
            headers = {
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize"
            }
            # Using data=f to stream upload
            response = robust_request("PUT", upload_url, headers=headers, data=f)
            response.raise_for_status()
            
        file_info = response.json()
        return file_info["file"]
        
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"Request failed: {e.response.status_code} {e.response.text}")
        raise e

def wait_for_processing(api_key, file_name):
    # file_name already contains 'files/', so don't append it again
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}"
    
    while True:
        try:
            response = robust_request("GET", url)
            response.raise_for_status()
            file_meta = response.json()
            state = file_meta.get("state")
            
            if state == "ACTIVE":
                return file_meta
            elif state == "FAILED":
                raise Exception("File processing failed state")
            
            print(".", end="", flush=True)
            time.sleep(2)
        except Exception as e:
            raise e

def get_best_model(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = robust_request("GET", url)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        
        # Debug: Print all available models to help troubleshooting
        available_names = [m['name'] for m in models]
        print(f"DEBUG: Available models from API: {available_names}")

        # Priority list - prefer newest models first as requested
        priorities = [
            "gemini-2.5-flash", "gemini-3.0-flash", "gemini-2.5-pro", "gemini-2.5",
            "gemini-3.0-pro", "gemini-3.0", 
            "gemini-2.0-flash", "gemini-2.0-pro", 
            "gemini-1.5-flash", "gemini-1.5-pro",
            "gemini-flash", "gemini-pro"
        ]
        
        for p in priorities:
            for m in models:
                if p in m["name"]:
                    print(f"Selected model: {m['name']}")
                    return m["name"] 
        
        # Fallback to first available gen model
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(f"Selected fallback model: {m['name']}")
                return m["name"]
                
        return "models/gemini-1.5-flash" # Default fallback
    except Exception as e:
        print(f"Warning: Could not list models ({e}). Using default.")
        return "models/gemini-1.5-flash"

def generate_content(api_key, file_uri, mime_type, model_name):
    # model_name already includes 'models/' prefix usually, but let's be safe
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
        
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    
    prompt = """Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[SPEECH]: <Word-for-word transcription. If none, write 'None'>
[SOUNDS]: <Description of sounds. If none, write 'None'>
[TEXT]: <Any on-screen text. If none, write 'None'>"""

    data = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
            ]
        }]
    }
    
    response = robust_request("POST", url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
        return ""

def main():
    if len(sys.argv) < 2:
        print("Usage: python rest_caption.py <directory_path> [api_key]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    api_key = get_api_key()
    output_file = input_dir / "captions.json"
    
    # Load existing
    existing_captions = {}
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    existing_captions[item.get('media_path')] = item.get('caption')
        except:
            pass

    # Find videos
    extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm']
    video_files = []
    for ext in extensions:
        video_files.extend(list(input_dir.glob(ext)))
    video_files.sort()
    
    print(f"Found {len(video_files)} videos.")
    
    # Get model once
    model_name = get_best_model(api_key)
    
    total = len(video_files)
    
    for i, video_path in enumerate(video_files):
        rel_path = video_path.name
        if rel_path in existing_captions:
            print(f"[{i+1}/{total}] Skipping {rel_path} (Already captioned)")
            continue
            
        print(f"[{i+1}/{total}] Processing {rel_path}...")
        try:
            print("  - Uploading...", end="\r")
            file_meta = upload_file(api_key, video_path)
            
            print("  - Processing... ", end="\r")
            wait_for_processing(api_key, file_meta["name"])
            
            print("  - Generating... ", end="\r")
            caption = generate_content(api_key, file_meta["uri"], file_meta["mimeType"], model_name)
            
            existing_captions[rel_path] = caption
            
            # Save
            captions_list = [{"media_path": k, "caption": v} for k, v in existing_captions.items()]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(captions_list, f, indent=2, ensure_ascii=False)
            
            print(f"  - Done.")
            time.sleep(10) # Throttle to avoid rate limits
            
        except Exception as e:
            print(f"\n  - Error: {e}")

if __name__ == "__main__":
    main()
