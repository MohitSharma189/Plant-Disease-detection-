from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import settings
from .preprocessing.quality import InvalidImageError, decode_image
from .schemas.analysis import (
    ChatRequest,
    ChatResponse,
    ChatStatusResponse,
    ClassResponse,
    HealthResponse,
    LearningFeedbackRequest,
    LearningFeedbackResponse,
    QualityResponse,
    ResearchRequest,
    ResearchResponse,
)
from .services.chat import ChatNotConfiguredError, ChatProviderError, answer_question, chat_provider_status
from .services.models import ModelUnavailableError, model_registry
from .services import learning as learning_service
from .services.research import ResearchProviderError, search_publications
from .services.screening import screen_plant_image


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_registry.load()
    yield


app = FastAPI(
    title=settings.app_name,
    description="CPU-first plant disease and visible affected-area analysis API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, object]:
    """Return operational and checkpoint status without claiming untrained models."""
    if (
        settings.model_dir != settings.model_dir.parent / "saved_models"
        or not getattr(settings, "classifier_path", None)
        or not settings.classifier_path.exists()
    ):
        classifier_checkpoint = settings.model_dir / "classifier.pt"
    else:
        classifier_checkpoint = settings.classifier_path
    segmentation_checkpoint = settings.model_dir / "segmentation.pt"
    model_health = model_registry.health()
    return {
        "status": "ok",
        "models_loaded": all(model_health.values()),
        "inference_ready": model_health["classifier_loaded"],
        **model_health,
        "classifier_checkpoint_present": classifier_checkpoint.exists(),
        "segmentation_checkpoint_present": segmentation_checkpoint.exists(),
        "model_load_error": model_registry.model_load_error,
        "segmentation_load_error": model_registry.segmentation_load_error,
        "message": (
            "Ready for classification and segmentation."
            if model_health["classifier_loaded"] and model_health["segmentation_loaded"]
            else "Classifier ready; segmentation is unavailable, so exact lesion masks and affected-area estimates are not produced."
            if model_health["classifier_loaded"]
            else f"Classifier is unavailable: {model_registry.model_load_error}"
            if model_registry.model_load_error
            else "A trained classifier checkpoint is required for inference."
        ),
    }


@app.get("/classes", response_model=ClassResponse)
def classes() -> dict[str, object]:
    """Return the class mapping saved by the training workflow."""
    plant_names = sorted({
        class_name.partition("___")[0].replace("_", " ").replace(",", "").strip()
        for class_name in (model_registry.class_names or [])
    })
    return {
        "supported_plant": ", ".join(plant_names) if plant_names else "",
        "classes": model_registry.class_names or [],
        "status": "ready" if model_registry.class_names else "not_configured",
        "message": (
            "Classes loaded from the trained model mapping."
            if model_registry.class_names
            else "Classes will be discovered during dataset preparation."
        ),
    }


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> dict[str, object]:
    try:
        return await search_publications(request.crop, request.disease, request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ResearchProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/chat/status", response_model=ChatStatusResponse)
async def chat_status() -> dict[str, object]:
    return await chat_provider_status()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> dict[str, object]:
    try:
        return await answer_question(request)
    except ChatNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ChatProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/learning/status")
def learning_status() -> dict[str, object]:
    return learning_service.learning_status()


@app.post("/learning/feedback", response_model=LearningFeedbackResponse)
def learning_feedback(request: LearningFeedbackRequest) -> dict[str, object]:
    accepted = learning_service.record_feedback(request.learning_id, request.helpful)
    return {
        "accepted": accepted,
        "message": "Thanks. Your rating updated response-style preferences." if accepted else "This rating was already used or has expired.",
    }


@app.delete("/learning/memory")
def clear_learning_memory() -> dict[str, bool]:
    learning_service.clear_learning()
    return {"cleared": True}


@app.get("/.well-known/agent-card.json")
async def a2a_agent_card(request: Request) -> dict[str, object]:
    """Publish the A2A agent card for clients that discover agents by URL."""
    return {
        "protocolVersion": "0.3.0",
        "name": settings.app_name,
        "description": "Multilingual plant-health and general question-answering assistant.",
        "url": f"{str(request.base_url).rstrip('/')}/a2a",
        "version": app.version,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [
            {
                "id": "plant-health-and-general-qa",
                "name": "Plant health and multilingual Q&A",
                "description": "Answers multilingual questions and provides cautious plant-health guidance.",
                "tags": ["plant-health", "multilingual", "question-answering"],
                "examples": ["How should I inspect an apple leaf with dark spots?", "What is 12 + 8 * 3?"],
                "inputModes": ["text"],
                "outputModes": ["text"],
            }
        ],
    }


@app.post("/a2a")
async def a2a_json_rpc(payload: Any = Body(...)) -> dict[str, object]:
    """Handle the A2A JSON-RPC message/send method as a one-shot task."""
    if not isinstance(payload, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid JSON-RPC request."},
        }
    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0" or not isinstance(payload.get("method"), str):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32600, "message": "Invalid JSON-RPC request."},
        }
    if payload["method"] != "message/send":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found."},
        }

    params = payload.get("params")
    message = params.get("message") if isinstance(params, dict) else None
    parts = message.get("parts") if isinstance(message, dict) else None
    if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(parts, list):
        question = ""
    else:
        question = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and part.get("kind") == "text"
            and isinstance(part.get("text"), str)
        ).strip()
    if not question:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "message.parts must include at least one text part."},
        }

    metadata = message.get("metadata", {}) if isinstance(message, dict) else {}
    try:
        language = None
        if isinstance(metadata, dict):
            language = metadata.get("response_language", metadata.get("responseLanguage"))
        chat_request = ChatRequest(question=question, response_language=language)
    except ValidationError:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "The message text or response language is invalid or too long."},
        }

    try:
        answer = await answer_question(chat_request)
    except ChatNotConfiguredError as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32003, "message": str(exc)},
        }
    except ChatProviderError as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }

    artifact: dict[str, object] = {
        "artifactId": str(uuid4()),
        "parts": [{"kind": "text", "text": answer["answer"]}],
    }
    if answer.get("learning_id"):
        artifact["metadata"] = {
            "learning_id": answer["learning_id"],
            "learning_category": answer.get("learning_category"),
        }
    task_id = str(uuid4())
    context_id = message.get("contextId") if isinstance(message.get("contextId"), str) else str(uuid4())
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "kind": "task",
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "completed", "timestamp": datetime.now(timezone.utc).isoformat()},
            "artifacts": [artifact],
        },
    }


@app.post("/quality", response_model=QualityResponse)
async def quality(file: UploadFile = File(...)) -> QualityResponse:
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        _, result = decode_image(contents)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return QualityResponse(**result.__dict__)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        image, quality_result = decode_image(contents)
        prediction = model_registry.predict_class(image)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"quality": quality_result.__dict__, "model": prediction}


@app.post("/screen-plant")
async def screen_plant(
    file: UploadFile = File(...),
    mode: str = Form(default="whole_plant"),
) -> dict[str, object]:
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        image, quality_result = decode_image(contents)
        screening = screen_plant_image(image, model_registry, mode)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"quality": quality_result.__dict__, "screening": screening}
