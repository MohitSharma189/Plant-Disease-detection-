from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    inference_ready: bool
    classifier_loaded: bool
    segmentation_loaded: bool
    classifier_checkpoint_present: bool
    segmentation_checkpoint_present: bool
    message: str
    model_load_error: str | None = None
    segmentation_load_error: str | None = None


class QualityResponse(BaseModel):
    width: int
    height: int
    format: str
    blur_score: float
    is_blurry: bool


class ClassResponse(BaseModel):
    supported_plant: str
    classes: list[str]
    status: str
    message: str


class ResearchRequest(BaseModel):
    crop: str = Field(..., min_length=1, max_length=100)
    disease: str = Field(..., min_length=1, max_length=120)
    max_results: int = Field(default=5, ge=1, le=10)


class ResearchArticle(BaseModel):
    title: str
    authors: str
    journal: str
    year: str
    doi: str | None = None
    url: str
    abstract: str | None = None


class ResearchResponse(BaseModel):
    provider: str
    query: str
    status: str
    searched_at: str
    results: list[ResearchArticle]
    message: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., min_length=1, max_length=1500)


class ChatSource(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: HttpUrl


class ChatContext(BaseModel):
    predicted_class: str | None = Field(default=None, max_length=200)
    model_score: float | None = Field(default=None, ge=0, le=1)
    score_calibrated: bool = False
    review_required: bool = False
    quality_note: str | None = Field(default=None, max_length=500)
    visible_evidence: str | None = Field(default=None, max_length=1000)
    limitations: list[str] = Field(default_factory=list, max_length=10)
    sources: list[ChatSource] = Field(default_factory=list, max_length=10)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1500)
    response_language: str | None = Field(default=None, max_length=100)
    assistant_notes: str = Field(default="", max_length=1200)
    context: ChatContext = Field(default_factory=ChatContext)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    citations: list[ChatSource]
    learning_id: str | None = None
    learning_category: str | None = None


class LearningFeedbackRequest(BaseModel):
    learning_id: str = Field(..., min_length=1, max_length=64)
    helpful: bool


class LearningFeedbackResponse(BaseModel):
    accepted: bool
    message: str


class ChatStatusResponse(BaseModel):
    configured: bool
    ready: bool
    model: str | None = None
    message: str


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Actionable error message.")


class PredictionResult(BaseModel):
    predicted_class: str = Field(..., description="Raw internal class identifier.")
    display_label: str = Field(..., description="Human-readable crop and condition label.")
    crop_name: str = Field(..., description="Human-readable crop name.")
    condition_name: str = Field(..., description="Human-readable condition or disease name.")
    is_healthy: bool = Field(..., description="True if the predicted condition is healthy.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top predicted score.")
    score_label: str = Field(..., description="Explicit label: Uncalibrated Model Score.")
    needs_review: bool = Field(..., description="True if score is below the operational review threshold.")
    is_uncertain: bool = Field(..., description="True if prediction falls into the uncertain prediction state.")
    review_threshold: float = Field(..., description="Operational review threshold (0.60).")
    confidence_calibrated: bool = Field(default=False, description="Whether temperature scaling was applied.")
    prediction_state: str = Field(..., description="Formally enumerated state: HIGH_CONFIDENCE or LOW_CONFIDENCE_UNCERTAIN.")
    prediction_status: str = Field(..., description="Status title: Model Prediction or Uncertain Prediction.")
    state_tone: str = Field(..., description="UI tone indicator: good or amber.")
    headline: str = Field(..., description="Concise summary headline.")
    explanation: str = Field(..., description="Scientific context and uncertainty explanation.")
    recommendation: str = Field(..., description="Actionable human-in-the-loop review recommendation.")
    class_scores: dict[str, float] | None = Field(default=None, description="Score distribution across classes.")


class CandidateRegion(BaseModel):
    box: dict[str, int]
    image: str
    predicted_class: str
    display_label: str
    crop_name: str
    condition_name: str
    confidence: float
    score_label: str
    confidence_calibrated: bool
    needs_review: bool
    explanation: str
    visible_evidence: str
    possible_causes: list[str]
    severity: str
    potential_impact: str
    immediate_actions: list[str]
    biological_treatment: str
    chemical_treatment: str
    sources: list[dict[str, str]]


class ScreeningResult(BaseModel):
    mode: str
    regions_checked: int
    candidate_regions: list[CandidateRegion]
    candidate_count: int
    overlay: str
    overall_prediction: PredictionResult | None = None
    explainability: dict[str, Any] | None = None
    candidate_score_threshold: float
    confidence_calibrated: bool
    review_threshold: float
    review_required: bool
    unknown_detection_validated: bool
    limitations: list[str]
    sources: list[dict[str, str]]
    professional_attribution: str
    screening_state: str = Field(..., description="Formally enumerated state: NO_CANDIDATES_IDENTIFIED or CANDIDATES_IDENTIFIED.")
    screening_status: str = Field(..., description="Screening status title.")
    screening_tone: str = Field(..., description="UI tone indicator: neutral or info.")
    screening_headline: str = Field(..., description="Headline describing screening outcome.")
    screening_message: str = Field(..., description="Detailed explanation of examined candidate windows.")
    safety_disclaimer: str = Field(..., description="Explicit warning that absence of candidates does not mean healthy.")
    score_label: str = Field(default="Uncalibrated Model Score")


class PredictResponse(BaseModel):
    quality: QualityResponse
    model: PredictionResult


class ScreenPlantResponse(BaseModel):
    quality: QualityResponse
    screening: ScreeningResult

