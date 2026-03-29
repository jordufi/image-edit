from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import websocket
import requests
import uuid
import json
import os
from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ComfyUI Qwen Image API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_comfyui_address():
    """Get ComfyUI address from environment variable or config file."""
    env_address = os.getenv("COMFYUI_ADDRESS")
    if env_address:
        return env_address
    config = load_config()
    return config["comfyui_backend"]


def establish_connection(server_address):
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    return ws, client_id


def send_workflow(server_address, client_id, workflow):
    payload = {"prompt": workflow, "client_id": client_id}
    return requests.post(f"http://{server_address}/prompt", json=payload)


def download_image(server_address, filename, subfolder, output_dir):
    url = f"http://{server_address}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": "output"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    return None


def update_qwen_prompt(workflow, prompt, negative_text, steps, cfg, image_a, seed=None, image_b=None, image_c=None):
    # Update prompts
    workflow["115:111"]["inputs"]["prompt"] = prompt
    workflow["115:110"]["inputs"]["prompt"] = negative_text

    # Update sampler parameters
    workflow["115:3"]["inputs"]["steps"] = steps
    workflow["115:3"]["inputs"]["cfg"] = cfg

    # Update primary image
    workflow["78"]["inputs"]["image"] = image_a

    # Update secondary image if present in workflow
    if image_b is not None and "120" in workflow:
        workflow["120"]["inputs"]["image"] = image_b

    # Update tertiary image if present in workflow
    if image_c is not None and "121" in workflow:
        workflow["121"]["inputs"]["image"] = image_c

    # Update seed
    workflow["115:3"]["inputs"]["seed"] = seed if seed is not None else 0

    return workflow


def get_history(server, prompt_id):
    """Get the execution history for a prompt"""
    try:
        response = requests.get(f"http://{server}/history/{prompt_id}")
        if response.status_code == 200:
            return response.json().get(prompt_id)
    except:
        pass
    return None


WORKFLOW_FILES = {
    1: "qwen-image.json",
    2: "qwen-2images.json",
    3: "qwen-3images.json",
}


class GenerateRequest(BaseModel):
    mode: int = 1
    prompt: str = "change this to a fire type pokemon but keeping his hat and cowboy jumpsuit and gloves"
    negative_prompt: str = ""
    steps: int = 4
    cfg: float = 1.0
    primary_image: str
    secondary_image: Optional[str] = None
    tertiary_image: Optional[str] = None
    seed: int = 0


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to ComfyUI and return the filename for use in generation."""
    server = get_comfyui_address()

    contents = await file.read()

    upload_url = f"http://{server}/upload/image"
    files = {'image': (file.filename, contents, file.content_type)}

    try:
        response = requests.post(upload_url, files=files)
        if response.status_code == 200:
            result = response.json()
            filename = result.get('name', file.filename)
            return {
                "status": "success",
                "filename": filename,
                "message": "Image uploaded successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"ComfyUI upload failed: {response.text}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload error: {str(e)}"
        )


@app.post("/generate")
def generate_image(request: GenerateRequest):
    """Generate an edited image with ComfyUI Qwen. Supports 1, 2, or 3 input images."""
    config = load_config()
    server = get_comfyui_address()

    if request.mode not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Mode must be 1, 2, or 3")
    if request.mode >= 2 and not request.secondary_image:
        raise HTTPException(status_code=400, detail="Secondary image required for 2-image mode")
    if request.mode >= 3 and not request.tertiary_image:
        raise HTTPException(status_code=400, detail="Tertiary image required for 3-image mode")

    workflow_file = WORKFLOW_FILES[request.mode]

    try:
        ws, client_id = establish_connection(server)

        with open(workflow_file, "r") as f:
            workflow = json.load(f)

        workflow = update_qwen_prompt(
            workflow,
            request.prompt,
            request.negative_prompt,
            request.steps,
            request.cfg,
            request.primary_image,
            request.seed,
            request.secondary_image,
            request.tertiary_image
        )

        r = send_workflow(server, client_id, workflow)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error sending workflow to ComfyUI: {r.text}")

        prompt_id = r.json().get('prompt_id')
        if not prompt_id:
            raise HTTPException(status_code=500, detail="No prompt ID received from ComfyUI")

        output_path = None

        while True:
            msg = ws.recv()
            if not msg:
                continue

            data = json.loads(msg)

            if data.get("type") == "execution_error":
                error_data = data.get("data", {})
                error_message = error_data.get("exception_message", "Unknown execution error")
                raise HTTPException(status_code=500, detail=f"ComfyUI execution error: {error_message}")

            if data.get("type") == "execution_start":
                continue

            if data.get("type") == "executed":
                node_id = data.get("data", {}).get("node")

                if node_id in ["60", "115:116"]:
                    output_data = data.get("data", {}).get("output", {})
                    images = output_data.get("images", [])

                    if images:
                        img = images[0]
                        filename = img["filename"]
                        subfolder = img.get("subfolder", "")

                        output_path = download_image(server, filename, subfolder, config["output_dir"])
                        if output_path:
                            break
                        else:
                            history = get_history(server, prompt_id)
                            if history and history.get("outputs"):
                                for node_output in history["outputs"].values():
                                    if "images" in node_output:
                                        for img in node_output["images"]:
                                            output_path = download_image(server, img["filename"], img.get("subfolder", ""), config["output_dir"])
                                            if output_path:
                                                break
                                    if output_path:
                                        break

                            if not output_path:
                                raise HTTPException(status_code=500, detail="Could not retrieve generated image")
                    else:
                        continue

        if not output_path:
            raise HTTPException(status_code=500, detail="Failed to download image")

        return FileResponse(output_path, media_type="image/png")

    except HTTPException:
        raise
    except websocket.WebSocketException as e:
        raise HTTPException(status_code=500, detail=f"WebSocket error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "ComfyUI Qwen Image Editing API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
