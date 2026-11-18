import { useEffect, useMemo, useRef, useState } from "react";
import { DiseaseInformationSection } from "./components/DiseaseInformationSection";

const API_URL = "http://127.0.0.1:8000";
const HISTORY_DB = "verdant-analysis-reports";
const HISTORY_STORE = "reports";
const AGENT_MEMORY_KEY = "verdant-agent-memory";
const AGENT_NOTES_KEY = "verdant-agent-notes";

function formatDisplayLabel(raw) {
  if (!raw) return "Unknown specimen";
  const parts = raw.split("___");
  const plant = parts[0] || "";
  const condition = parts[1] || "";
  let crop = plant.replace(/_/g, " ").replace(/,/g, "").trim();
  if (crop.toLowerCase().includes("cherry")) {
    crop = crop.toLowerCase().includes("sour") ? "Cherry (Sour)" : "Cherry";
  } else if (crop.toLowerCase().includes("pepper") && crop.toLowerCase().includes("bell")) {
    crop = "Bell Pepper";
  } else if (crop.toLowerCase().includes("corn")) {
    crop = "Corn (Maize)";
  } else {
    crop = crop.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  let cond = condition.replace(/_/g, " ").trim();
  if (raw.toLowerCase().includes("gray_leaf_spot") && raw.toLowerCase().includes("cercospora")) {
    cond = "Cercospora Leaf Spot / Gray Leaf Spot";
  } else if (raw.toLowerCase().includes("two-spotted") || raw.toLowerCase().includes("spider_mites")) {
    cond = "Spider Mites (Two-Spotted Spider Mite)";
  } else if (raw.toLowerCase().includes("haunglongbing") || raw.toLowerCase().includes("citrus_greening")) {
    cond = "Huanglongbing (Citrus Greening)";
  } else if (raw.toLowerCase().includes("esca")) {
    cond = "Esca (Black Measles)";
  } else if (raw.toLowerCase().includes("isariopsis")) {
    cond = "Leaf Blight (Isariopsis Leaf Spot)";
  } else if (raw.toLowerCase().includes("yellow_leaf_curl")) {
    cond = "Yellow Leaf Curl Virus";
  } else if (raw.toLowerCase().includes("mosaic_virus")) {
    cond = "Mosaic Virus";
  } else if (cond.toLowerCase() === "healthy") {
    cond = "Healthy";
  } else {
    cond = cond.replace(/\b\w/g, (c) => c.toUpperCase());
  }
  return `${crop} - ${cond}`;
}

function stripSafetyNoticePrefix(text) {
  if (!text) return "";
  return String(text).replace(/^(?:important\s+safety\s+notice\s*:\s*)+/i, "").trim();
}

function formatScorePercent(score) {
  const numeric = Number(score);
  if (!Number.isFinite(numeric)) return "0.0%";
  return `${(numeric * 100).toFixed(1)}%`;
}

function openHistoryDb() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("This browser does not support saved analysis details."));
      return;
    }
    const request = window.indexedDB.open(HISTORY_DB, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(HISTORY_STORE)) {
        database.createObjectStore(HISTORY_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open saved analysis storage."));
  });
}

async function saveHistoryDetail(record) {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    transaction.objectStore(HISTORY_STORE).put(record);
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not save analysis details."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Saving analysis details was interrupted."));
    };
  });
}

async function saveHistoryDetails(records) {
  if (!records.length) return;
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    const store = transaction.objectStore(HISTORY_STORE);
    records.forEach((record) => store.put(record));
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not migrate saved history."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Migrating saved history was interrupted."));
    };
  });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read an image for export."));
    reader.readAsDataURL(blob);
  });
}

async function loadHistoryDetail(id) {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readonly");
    const request = transaction.objectStore(HISTORY_STORE).get(id);
    request.onsuccess = () => {
      database.close();
      resolve(request.result || null);
    };
    request.onerror = () => {
      database.close();
      reject(request.error || new Error("Could not load saved analysis details."));
    };
  });
}

async function listHistoryDetails() {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readonly");
    const request = transaction.objectStore(HISTORY_STORE).getAll();
    request.onsuccess = () => resolve(request.result || []);
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not load saved history."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Loading saved history was interrupted."));
    };
  });
}

async function deleteHistoryDetail(id) {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    transaction.objectStore(HISTORY_STORE).delete(id);
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not delete this saved analysis."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Deleting this saved analysis was interrupted."));
    };
  });
}

async function clearHistoryDetails() {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    transaction.objectStore(HISTORY_STORE).clear();
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not clear saved history."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Clearing saved history was interrupted."));
    };
  });
}

function readLegacyHistory() {
  try {
    const records = JSON.parse(localStorage.getItem("verdant-analysis-history") || "[]");
    return Array.isArray(records) ? records : [];
  } catch {
    return [];
  }
}

function speechLocaleFor() {
  return navigator.languages?.[0] || navigator.language || "en-US";
}

function Icon({ name }) {
  const paths = {
    grid: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
    plus: "M12 5v14M5 12h14",
    history: "M4 12a8 8 0 1 0 2.34-5.66L4 8.68M4 4v4.68h4.68M12 8v4l2.5 2",
    arrow: "M5 12h13M13 6l6 6-6 6",
    upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
    leaf: "M20 4C10 4 5 8 5 15c0 3 2 5 5 5 6 0 10-6 10-16ZM5 20c1-5 4-8 9-10",
    check: "m5 12 4 4L19 6",
    mic: "M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3ZM5 10v1a7 7 0 0 0 14 0v-1M12 18v4m-4 0h8",
    speaker: "M11 5 6 9H3v6h3l5 4V5Zm4 4a5 5 0 0 1 0 6m3-9a9 9 0 0 1 0 12",
    stop: "M6 6h12v12H6z",
  };
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <path d={paths[name]} />
    </svg>
  );
}

function StatusDot({ tone = "neutral" }) {
  return <span className={`status-dot ${tone}`} aria-hidden="true" />;
}

function ReadinessCard({ label, value, detail, ready, icon }) {
  return (
    <article className="readiness-card">
      <div className="card-icon"><Icon name={icon} /></div>
      <div className="card-copy">
        <p className="meta-label">{label}</p>
        <strong>{value}</strong>
        <p>{detail}</p>
      </div>
      <StatusDot tone={ready ? "good" : "warn"} />
    </article>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [classes, setClasses] = useState(null);
  const [error, setError] = useState("");
  const [view, setView] = useState("overview");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [analysisState, setAnalysisState] = useState("idle");
  const [analysisError, setAnalysisError] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [fetchedDiseaseInfo, setFetchedDiseaseInfo] = useState(null);
  const [researchState, setResearchState] = useState("idle");
  const [researchResult, setResearchResult] = useState(null);
  const [researchError, setResearchError] = useState("");
  const [historyError, setHistoryError] = useState("");
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [openingHistoryId, setOpeningHistoryId] = useState(null);
  const [deletingHistoryId, setDeletingHistoryId] = useState(null);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [exportingHistory, setExportingHistory] = useState(false);
  const [historyRestoreId, setHistoryRestoreId] = useState(null);
  const [saveHistoryConsent, setSaveHistoryConsent] = useState(false);
  const defaultWelcomeText = "I'm your plant-health assistant. I can help explain your specimen's predicted class, model score, screening results, uncertainty, and recommended next steps.";
  const defaultConversation = [
    {
      id: "welcome-message",
      role: "assistant",
      text: defaultWelcomeText,
      citations: [],
    },
  ];
  const [conversation, setConversation] = useState(defaultConversation);
  const [assistantNotes, setAssistantNotes] = useState(() => {
    try {
      return localStorage.getItem(AGENT_NOTES_KEY) || "";
    } catch {
      return "";
    }
  });
  const [activeHistoryRecordId, setActiveHistoryRecordId] = useState(null);
  const [chatQuestion, setChatQuestion] = useState("");

  const activeDiseaseInfo = useMemo(() => {
    return analysisResult?.screening?.disease_info || analysisResult?.disease_info || fetchedDiseaseInfo;
  }, [analysisResult, fetchedDiseaseInfo]);

  useEffect(() => {
    const leadClass =
      analysisResult?.screening?.overall_prediction?.predicted_class ||
      analysisResult?.screening?.candidate_regions?.[0]?.predicted_class ||
      analysisResult?.model?.predicted_class;

    if (leadClass && !analysisResult?.screening?.disease_info && !analysisResult?.disease_info) {
      let active = true;
      fetch(`${API_URL}/disease-info/${encodeURIComponent(leadClass)}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (active && data) setFetchedDiseaseInfo(data);
        })
        .catch(() => {});
      return () => {
        active = false;
      };
    } else {
      setFetchedDiseaseInfo(null);
    }
  }, [analysisResult]);

  useEffect(() => {
    try {
      const storedConversation = JSON.parse(localStorage.getItem(AGENT_MEMORY_KEY) || "null");
      if (Array.isArray(storedConversation) && storedConversation.length) {
        setConversation(
          storedConversation.map((message) => (
            message?.id === "welcome-message"
              ? { ...message, text: defaultWelcomeText }
              : message
          )),
        );
      }
    } catch {
      setConversation(defaultConversation);
    }
  }, []);

  useEffect(() => {
    if (assistantNotes.trim()) {
      localStorage.setItem(AGENT_NOTES_KEY, assistantNotes.slice(0, 1200));
    } else {
      localStorage.removeItem(AGENT_NOTES_KEY);
    }
  }, [assistantNotes]);

  useEffect(() => {
    if (conversation.length > 0) {
      localStorage.setItem(AGENT_MEMORY_KEY, JSON.stringify(conversation));
    } else {
      localStorage.removeItem(AGENT_MEMORY_KEY);
    }
  }, [conversation]);
  const [chatError, setChatError] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const [chatStatus, setChatStatus] = useState(null);
  const [learningStatus, setLearningStatus] = useState(null);
  const [learningSaving, setLearningSaving] = useState(false);
  const [voiceListening, setVoiceListening] = useState(false);
  const [speechPlaying, setSpeechPlaying] = useState(false);
  const [analysisMode, setAnalysisMode] = useState("leaf");
  const [history, setHistory] = useState(() => {
    return readLegacyHistory();
  });
  const fileInput = useRef(null);
  const cameraVideo = useRef(null);
  const cameraStream = useRef(null);
  const currentView = useRef(view);
  const speechRecognition = useRef(null);
  const researchRequestId = useRef(0);
  const conversationEndRef = useRef(null);
  const chatSendingRef = useRef(false);
  const voiceInputSupported = typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/learning/status`)
      .then((response) => response.ok ? response.json() : null)
      .then((status) => {
        if (active && status) setLearningStatus(status);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    currentView.current = view;
  }, [view]);

  useEffect(() => {
    if (cameraActive && cameraVideo.current && cameraStream.current) {
      cameraVideo.current.srcObject = cameraStream.current;
      cameraVideo.current.play().catch((playError) => {
        setCameraError(`Could not start the camera preview: ${playError.message}`);
      });
    }
  }, [cameraActive]);

  useEffect(() => () => {
    cameraStream.current?.getTracks().forEach((track) => track.stop());
    cameraStream.current = null;
    speechRecognition.current?.stop();
    window.speechSynthesis?.cancel();
  }, []);

  useEffect(() => {
    if (view !== "analysis" && cameraStream.current) {
      cameraStream.current.getTracks().forEach((track) => track.stop());
      cameraStream.current = null;
      setCameraActive(false);
    }
  }, [view]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [conversation, chatSending]);

  useEffect(() => {
    let active = true;
    const restoreHistoryList = async () => {
      try {
        const [savedRecords, localRecords] = await Promise.all([
          listHistoryDetails(),
          Promise.resolve().then(readLegacyHistory),
        ]);
        if (!active) return;

        const localById = new Map(localRecords.map((record) => [String(record.id), record]));
        const savedIds = new Set(savedRecords.map((record) => String(record.id)));
        const mergedRecords = savedRecords.map((record) => ({
          ...localById.get(String(record.id)),
          ...record,
        }));
        const mergedIds = new Set(mergedRecords.map((record) => String(record.id)));
        const legacyOnlyRecords = localRecords.filter((record) => !mergedIds.has(String(record.id)));
        const allRecords = [...mergedRecords, ...legacyOnlyRecords]
          .sort((left, right) => new Date(right.createdAt || 0) - new Date(left.createdAt || 0));

        await saveHistoryDetails(
          allRecords.filter((record) => !savedIds.has(String(record.id))),
        );
        if (!active) return;
        setHistory(allRecords);
        localStorage.removeItem("verdant-analysis-history");
      } catch (loadError) {
        if (active) {
          setHistoryError(`Could not load the complete saved history: ${loadError.message}`);
        }
      } finally {
        if (active) setHistoryLoaded(true);
      }
    };
    restoreHistoryList();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    let retryTimer;
    const checkBackend = async () => {
      try {
        const [healthResponse, classesResponse, chatStatusResponse] = await Promise.all([
          fetch(`${API_URL}/health`).then((response) => {
            if (!response.ok) throw new Error(`Backend health check failed (${response.status}).`);
            return response.json();
          }),
          fetch(`${API_URL}/classes`).then((response) => {
            if (!response.ok) throw new Error(`Backend class check failed (${response.status}).`);
            return response.json();
          }),
          fetch(`${API_URL}/chat/status`).then((response) => {
            if (!response.ok) throw new Error(`Chat configuration check failed (${response.status}).`);
            return response.json();
          }),
        ]);
        if (!active) return;
        setHealth(healthResponse);
        setClasses(classesResponse);
        setChatStatus(chatStatusResponse);
        setError("");
      } catch (requestError) {
        if (active) {
          setError(`${requestError.message} Retrying automatically; make sure FastAPI is running at ${API_URL}.`);
        }
      } finally {
        if (active) retryTimer = window.setTimeout(checkBackend, 5000);
      }
    };
    checkBackend();
    const refreshOnReturn = () => {
      if (document.visibilityState === "visible") checkBackend();
    };
    window.addEventListener("focus", refreshOnReturn);
    document.addEventListener("visibilitychange", refreshOnReturn);
    return () => {
      active = false;
      window.clearTimeout(retryTimer);
      window.removeEventListener("focus", refreshOnReturn);
      document.removeEventListener("visibilitychange", refreshOnReturn);
    };
  }, []);

  const classifierReady = Boolean(health?.inference_ready ?? health?.classifier_loaded ?? health?.classifier_ready ?? health?.classifier_checkpoint_present);
  const segmentationReady = Boolean(health?.segmentation_loaded ?? health?.segmentation_ready ?? health?.segmentation_checkpoint_present);
  const modelsReady = classifierReady;
  const navItems = [
    { id: "overview", label: "Overview", icon: "grid" },
    { id: "analysis", label: "New analysis", icon: "plus" },
    { id: "history", label: "History", icon: "history" },
  ];

  const readinessLabel = health ? (modelsReady ? "Ready for inference" : "Model setup needed") : "Checking backend";
  const readinessTone = modelsReady ? "good" : health ? "warn" : "neutral";
  const classCount = classes?.classes?.length ?? 0;
  const classText = classes ? (classCount ? `${classCount} classes loaded` : "Class mapping not configured") : "Waiting for backend";
  const scoreLabel = analysisResult?.screening?.overall_prediction?.score_label
    || analysisResult?.screening?.score_label
    || (analysisResult?.screening?.confidence_calibrated
        ? "temperature-scaled score (PlantVillage calibration; not field calibration)"
        : "Uncalibrated Model Score");

  const stopCamera = () => {
    cameraStream.current?.getTracks().forEach((track) => track.stop());
    cameraStream.current = null;
    setCameraActive(false);
  };

  const useImageFile = (nextFile) => {
    if (!nextFile) return;
    const supportedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!supportedTypes.includes(nextFile.type)) {
      setFile(null);
      setPreview("");
      setAnalysisState("blocked");
      setAnalysisError("Choose a JPEG, PNG, or WEBP image.");
      return;
    }
    if (nextFile.size === 0) {
      setFile(null);
      setPreview("");
      setAnalysisState("blocked");
      setAnalysisError("The selected image file is empty (0 bytes).");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setFile(null);
      setPreview("");
      setAnalysisState("blocked");
      setAnalysisError("Image exceeds the 10 MB upload limit.");
      return;
    }

    const objectUrl = URL.createObjectURL(nextFile);
    const img = new Image();
    img.onload = () => {
      if (img.naturalWidth < 128 || img.naturalHeight < 128) {
        setFile(null);
        setPreview("");
        setAnalysisState("blocked");
        setAnalysisError(`Image dimensions (${img.naturalWidth}×${img.naturalHeight}px) must be at least 128×128 pixels.`);
      }
    };
    img.src = objectUrl;

    setFile(nextFile);
    setAnalysisError("");
    setAnalysisState("idle");
    setPreview(objectUrl);
  };

  const handleFile = (selected) => useImageFile(selected?.[0]);

  const startCamera = async () => {
    if (cameraStarting) return;
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access is not available in this browser or page context. Open the app on localhost or HTTPS and choose an image file instead.");
      return;
    }
    setCameraStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" } },
      });
      if (currentView.current !== "analysis") {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStream.current = stream;
      setCameraActive(true);
    } catch (cameraRequestError) {
      const message = cameraRequestError.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in your browser settings, or choose an image file."
        : cameraRequestError.name === "NotFoundError"
          ? "No camera was found on this device. Choose an image file instead."
          : `Could not access the camera: ${cameraRequestError.message}`;
      setCameraError(message);
    } finally {
      setCameraStarting(false);
    }
  };

  const captureCameraImage = () => {
    const video = cameraVideo.current;
    if (!video?.videoWidth || !video.videoHeight) {
      setCameraError("The camera is still starting. Wait for the preview, then try capture again.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("Could not capture the camera image. Please try again.");
        return;
      }
      const capturedFile = new File([blob], `plant-camera-${new Date().toISOString().replaceAll(":", "-")}.jpg`, { type: "image/jpeg" });
      useImageFile(capturedFile);
      setCameraError("");
      stopCamera();
    }, "image/jpeg", 0.92);
  };

  const submitChatQuestion = async (event) => {
    event.preventDefault();
    const text = chatQuestion.trim();
    if (!text || chatSending || chatSendingRef.current) return;
    if (chatStatus?.configured && !chatStatus.ready) {
      setChatError("The assistant is temporarily unavailable. Please try again shortly.");
      return;
    }
    const assistantId = window.crypto?.randomUUID?.() || `${Date.now()}-assistant`;
    const userId = window.crypto?.randomUUID?.() || `${Date.now()}-user`;
    const nextUserMessage = { id: userId, role: "user", text };
    const assistantPlaceholder = {
      id: assistantId,
      role: "assistant",
      text: "",
      citations: [],
      isStreaming: true,
    };

    chatSendingRef.current = true;
    setChatSending(true);
    setChatError("");
    setConversation((currentConversation) => [...currentConversation, nextUserMessage, assistantPlaceholder]);
    setChatQuestion("");

    try {
      const screening = analysisResult?.screening || {};
      const overall = screening.overall_prediction || null;
      const candidateList = Array.isArray(screening.candidate_regions) ? screening.candidate_regions : [];
      const lead = overall || candidateList[0] || null;
      const reviewThreshold = typeof overall?.review_threshold === "number"
        ? overall.review_threshold
        : (typeof screening.review_threshold === "number" ? screening.review_threshold : 0.60);
      const candidateThreshold = typeof screening.candidate_score_threshold === "number"
        ? screening.candidate_score_threshold
        : 0.60;
      const stageAUncertain = overall
        ? Boolean(overall.is_uncertain || overall.needs_review || (typeof overall.confidence === "number" && overall.confidence < reviewThreshold))
        : null;
      const candidateCount = typeof screening.candidate_count === "number"
        ? screening.candidate_count
        : (analysisResult?.screening ? candidateList.length : null);
      const isInconclusive = analysisResult?.screening
        ? Boolean(stageAUncertain || (candidateCount === 0 && !overall))
        : null;
      const suppliedSources = [
        ...(researchResult?.results || []).map((source) => ({ title: source.title, url: source.url })),
        ...(lead?.sources || screening.sources || []).map((source) => ({
          title: source.title || source.name,
          url: source.url,
        })),
      ].filter((source, index, sources) => source.title && source.url && sources.findIndex((item) => item.url === source.url) === index).slice(0, 10);
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          assistant_notes: assistantNotes.trim(),
          context: {
            predicted_class: overall?.predicted_class || null,
            display_label: overall?.display_label || (overall?.predicted_class ? formatDisplayLabel(overall.predicted_class) : null),
            crop_name: overall?.crop_name || null,
            condition_name: overall?.condition_name || null,
            is_healthy: typeof overall?.is_healthy === "boolean" ? overall.is_healthy : null,
            model_score: typeof overall?.confidence === "number" ? overall.confidence : null,
            review_threshold: reviewThreshold,
            score_calibrated: Boolean(overall?.confidence_calibrated ?? screening.confidence_calibrated),
            review_required: Boolean(screening.review_required || overall?.needs_review),
            stage_a_uncertain: stageAUncertain,
            stage_a_status: overall?.prediction_state || overall?.prediction_status || null,
            screening_mode: screening.mode || null,
            regions_checked: typeof screening.regions_checked === "number" ? screening.regions_checked : null,
            candidate_count: candidateCount,
            candidate_score_threshold: candidateThreshold,
            candidate_regions: candidateList.slice(0, 8).map((region) => ({
              predicted_class: region.predicted_class,
              display_label: region.display_label || formatDisplayLabel(region.predicted_class),
              confidence: region.confidence,
              needs_review: Boolean(region.needs_review),
              box: region.box || null,
            })),
            is_inconclusive: isInconclusive,
            has_gradcam: Boolean(screening.explainability),
            quality_note: analysisResult?.quality?.is_blurry
              ? `Image may be blurry (Laplacian sharpness score ${analysisResult.quality.blur_score.toFixed(1)}).`
              : null,
            visible_evidence: lead?.visible_evidence || null,
            limitations: (screening.limitations || []).slice(0, 10),
            sources: suppliedSources,
            history: conversation.slice(-10).map(({ role, text: messageText }) => ({ role, text: messageText })),
          },
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The assistant could not complete your question.");

      const fullAnswer = payload.answer || "I'm ready to help. Ask a follow-up question about your specimen or model results.";
      const citations = payload.citations || [];
      setConversation((currentConversation) => currentConversation.map((message) => (
        message.id === assistantId
          ? {
            ...message,
            text: fullAnswer,
            citations,
            learningId: payload.learning_id || null,
            learningCategory: payload.learning_category || null,
            isStreaming: false,
          }
          : message
      )));

      if (activeHistoryRecordId) {
        try {
          const saved = await loadHistoryDetail(activeHistoryRecordId);
          if (saved) {
            const updatedConversation = [...conversation, nextUserMessage, { ...assistantPlaceholder, text: fullAnswer, citations, isStreaming: false }];
            await saveHistoryDetail({ ...saved, conversation: updatedConversation });
          }
        } catch (storageError) {
          setChatError(`Answer received, but conversation could not be saved: ${storageError.message}`);
        }
      }
    } catch (requestError) {
      setConversation((currentConversation) => currentConversation.map((message) => {
        if (message.id !== assistantId) return message;
        return {
          ...message,
          text: `${requestError.message}`,
          citations: [],
          isStreaming: false,
        };
      }));
      setChatError(requestError.message);
    } finally {
      chatSendingRef.current = false;
      setChatSending(false);
    }
  };

  const rateAssistantAnswer = async (messageId, helpful) => {
    const message = conversation.find((item) => item.id === messageId);
    if (!message?.learningId || message.learningFeedback != null || message.feedbackSubmitting) return;
    setConversation((currentConversation) => currentConversation.map((item) => (
      item.id === messageId ? { ...item, feedbackSubmitting: true } : item
    )));
    try {
      const response = await fetch(`${API_URL}/learning/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ learning_id: message.learningId, helpful }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result.accepted) throw new Error(result.message || "Could not save your rating.");
      setConversation((currentConversation) => currentConversation.map((item) => (
        item.id === messageId ? { ...item, feedbackSubmitting: false, learningFeedback: helpful } : item
      )));
      const statusResponse = await fetch(`${API_URL}/learning/status`);
      if (statusResponse.ok) setLearningStatus(await statusResponse.json());
    } catch (feedbackError) {
      setConversation((currentConversation) => currentConversation.map((item) => (
        item.id === messageId ? { ...item, feedbackSubmitting: false } : item
      )));
      setChatError(feedbackError.message);
    }
  };

  const resetAssistantLearning = async () => {
    if (!window.confirm("Clear all learned response-style preferences and unused ratings?")) return;
    setLearningSaving(true);
    try {
      const response = await fetch(`${API_URL}/learning/memory`, { method: "DELETE" });
      if (!response.ok) throw new Error("Could not reset learned response preferences.");
      setLearningStatus({ ratings: 0, categories: {} });
      setConversation((currentConversation) => currentConversation.map((message) => ({
        ...message,
        learningId: null,
        learningFeedback: null,
      })));
    } catch (resetError) {
      setChatError(resetError.message);
    } finally {
      setLearningSaving(false);
    }
  };

  const startVoiceInput = () => {
    if (voiceListening && speechRecognition.current) {
      speechRecognition.current.stop();
      setVoiceListening(false);
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setChatError("Voice input is not currently supported in this browser. Please type your question instead.");
      return;
    }
    setChatError("");
    const recognition = new SpeechRecognition();
    recognition.lang = chatStatus?.configured ? speechLocaleFor() : "en-US";
    recognition.interimResults = false;
    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      if (transcript) {
        setChatQuestion((current) => (current.trim() ? `${current.trim()} ${transcript}` : transcript));
      }
      setVoiceListening(false);
    };
    recognition.onerror = (event) => {
      setChatError(`Voice input could not complete (${event.error || "unknown error"}). You can type your question instead.`);
      setVoiceListening(false);
    };
    recognition.onend = () => setVoiceListening(false);
    speechRecognition.current = recognition;
    try {
      recognition.start();
      setVoiceListening(true);
    } catch (voiceError) {
      setChatError(`Could not start voice input: ${voiceError.message}`);
      setVoiceListening(false);
    }
  };

  const speakText = (text) => {
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
      setChatError("Spoken playback is not supported by this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new window.SpeechSynthesisUtterance(text);
    utterance.lang = speechLocaleFor();
    utterance.onend = () => setSpeechPlaying(false);
    utterance.onerror = () => {
      setSpeechPlaying(false);
      setChatError("Spoken playback failed. The text response is still available.");
    };
    setSpeechPlaying(true);
    window.speechSynthesis.speak(utterance);
  };

  const stopSpeech = () => {
    window.speechSynthesis?.cancel();
    setSpeechPlaying(false);
  };

  const researchLeadingCandidate = async (result) => {
    const requestId = ++researchRequestId.current;
    const screening = result?.screening;
    const label = screening?.candidate_regions?.[0]?.predicted_class
      || screening?.overall_prediction?.predicted_class;
    const [rawCrop = "", ...rawDiseaseParts] = String(label || "").split("___");
    const crop = rawCrop.replaceAll("_", " ").replaceAll(",", "").trim();
    const disease = rawDiseaseParts.join(" ").replaceAll("_", " ").trim();
    if (!crop || !disease || disease.toLowerCase() === "healthy") {
      if (requestId === researchRequestId.current) setResearchState("none");
      return;
    }

    setResearchState("loading");
    setResearchResult(null);
    setResearchError("");
    try {
      const response = await fetch(`${API_URL}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop, disease, max_results: 5 }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Scholarly search is unavailable.");
      if (requestId !== researchRequestId.current) return;
      setResearchResult(payload);
      setResearchState(payload.status === "results" ? "done" : "empty");
    } catch (requestError) {
      if (requestId !== researchRequestId.current) return;
      setResearchError(requestError.message);
      setResearchState("error");
    }
  };

  const runAnalysis = async () => {
    if (!file) return;
    if (!classifierReady) {
      setAnalysisState("blocked");
      setAnalysisError("Analysis is unavailable until a trained classifier checkpoint is installed.");
      return;
    }
    setAnalysisState("loading");
    researchRequestId.current += 1;
    setAnalysisError("");
    setAnalysisResult(null);
    setConversation(defaultConversation);
    setActiveHistoryRecordId(null);
    setChatError("");
    setResearchState("idle");
    setResearchResult(null);
    setResearchError("");
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("mode", analysisMode === "whole" ? "whole_plant" : "single_leaf");
      const response = await fetch(`${API_URL}/screen-plant`, { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Analysis endpoint is unavailable.");
      setAnalysisResult(payload);
      void researchLeadingCandidate(payload);
      const record = {
        id: historyRestoreId ?? (window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`),
        filename: file.name,
        predictedClass: `${payload.screening?.candidate_count || 0} candidate region${payload.screening?.candidate_count === 1 ? "" : "s"}`,
        confidence: payload.screening?.overall_prediction?.confidence ?? payload.screening?.candidate_regions?.[0]?.confidence,
        createdAt: new Date().toISOString(),
      };
      setHistoryError("");
      if (saveHistoryConsent) {
        try {
          await saveHistoryDetail({
            ...record,
            id: record.id,
            filename: file.name,
            imageType: file.type || "image/jpeg",
            imageBlob: file.slice(0, file.size, file.type || "image/jpeg"),
            result: payload,
            conversation: [],
          });
          setActiveHistoryRecordId(record.id);
          setHistory((previousHistory) => [record, ...previousHistory.filter((entry) => entry.id !== record.id)]);
        } catch (storageError) {
          setHistoryError(`Analysis finished, but this record could not be saved for later: ${storageError.message}`);
        }
      }
      setHistoryRestoreId(null);
      setAnalysisState("done");
    } catch (requestError) {
      setAnalysisState("error");
      setAnalysisError(requestError.message);
    }
  };

  const removeHistoryRecord = async (record) => {
    setDeletingHistoryId(record.id);
    setHistoryError("");
    try {
      await deleteHistoryDetail(record.id);
      setHistory((previousHistory) => previousHistory.filter((entry) => entry.id !== record.id));
      setHistoryRestoreId((currentId) => currentId === record.id ? null : currentId);
      localStorage.removeItem("verdant-analysis-history");
    } catch (deleteError) {
      setHistoryError(deleteError.message);
    } finally {
      setDeletingHistoryId(null);
    }
  };

  const removeAllHistory = async () => {
    if (!window.confirm("Delete all saved analysis history from this browser? This cannot be undone.")) return;
    setClearingHistory(true);
    setHistoryError("");
    try {
      await clearHistoryDetails();
      setHistory([]);
      setHistoryRestoreId(null);
      localStorage.removeItem("verdant-analysis-history");
    } catch (clearError) {
      setHistoryError(clearError.message);
    } finally {
      setClearingHistory(false);
    }
  };

  const exportHistory = async () => {
    setExportingHistory(true);
    setHistoryError("");
    try {
      const records = await Promise.all(history.map(async (summary) => {
        const detail = await loadHistoryDetail(summary.id);
        if (!detail) return summary;
        const { imageBlob, ...record } = detail;
        return {
          ...record,
          originalImage: imageBlob ? await blobToDataUrl(imageBlob) : null,
        };
      }));
      const exportFile = new Blob(
        [JSON.stringify({ exportedAt: new Date().toISOString(), records }, null, 2)],
        { type: "application/json" },
      );
      const url = window.URL.createObjectURL(exportFile);
      const link = document.createElement("a");
      link.href = url;
      link.download = `verdant-history-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (exportError) {
      setHistoryError(`Could not export saved history: ${exportError.message}`);
    } finally {
      setExportingHistory(false);
    }
  };

  const reopenHistoryRecord = async (record) => {
    setOpeningHistoryId(record.id);
    setHistoryError("");
    try {
      const saved = await loadHistoryDetail(record.id);
      if (!saved?.result || !saved?.imageBlob) {
        setHistoryRestoreId(record.id);
        setHistoryError(`“${record.filename}” was saved before full reports were available. Its original image and detailed result are not stored, so select that image and rerun the analysis to restore this entry.`);
        return;
      }
      const restoredFile = new File([saved.imageBlob], saved.filename, { type: saved.imageType || "image/jpeg" });
      setFile(restoredFile);
      setPreview(URL.createObjectURL(saved.imageBlob));
      setAnalysisResult(saved.result);
      setConversation(Array.isArray(saved.conversation) && saved.conversation.length ? saved.conversation : defaultConversation);
      setActiveHistoryRecordId(record.id);
      setAnalysisMode(saved.result.screening?.mode === "whole_plant" ? "whole" : "leaf");
      setAnalysisError("");
      setAnalysisState("done");
      setView("analysis");
    } catch (openError) {
      setHistoryError(openError.message);
    } finally {
      setOpeningHistoryId(null);
    }
  };

  const beginHistoryRestore = (recordId) => {
    setHistoryRestoreId(recordId);
    setFile(null);
    setPreview("");
    setAnalysisResult(null);
    setAnalysisError("");
    setAnalysisState("idle");
    setHistoryError("");
    setAnalysisMode("whole");
    setView("analysis");
    window.setTimeout(() => {
      if (fileInput.current) {
        fileInput.current.value = "";
        fileInput.current.click();
      }
    }, 0);
  };

  const subtitle = useMemo(() => ({
    overview: "A measured view of plant health, model readiness, and your next field note.",
    analysis: "Screen a single leaf or a whole plant. Review only candidate crops, with model and evidence limitations made explicit.",
    history: "Completed assessments saved in this browser, with nothing fabricated.",
  }[view]), [view]);

  const quickPrompts = [
    "Summarize this screening result and the main uncertainty.",
    "What does this uncalibrated model score mean?",
    "Explain the candidate-region screening outcome.",
    "How does whole-leaf classification differ from crop screening?",
    "What should I check in the field before taking action?",
  ];

  const handleSuggestedPrompt = (prompt) => setChatQuestion(prompt);
  const rememberUserMessage = (messageId) => {
    const message = conversation.find((item) => item.id === messageId);
    if (!message || message.role !== "user" || message.remembered) return;

    const note = `Saved from chat: ${message.text.trim()}`;
    setAssistantNotes((currentNotes) => {
      const entries = [...currentNotes.split("\n").filter(Boolean), note];
      while (entries.length > 1 && entries.join("\n").length > 1200) entries.shift();
      return entries.join("\n").slice(-1200);
    });
    setConversation((currentConversation) => currentConversation.map((item) => (
      item.id === messageId ? { ...item, remembered: true } : item
    )));
  };
  const agentMemoryLength = conversation.filter((message) => message.role === "user" || message.role === "assistant").length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Icon name="leaf" /></div>
          <div><strong>Verdant</strong><span>field laboratory</span></div>
        </div>
        <div className="side-rule" />
        <p className="meta-label nav-heading">Workspace</p>
        <nav aria-label="Primary navigation">
          {navItems.map((item) => (
            <button key={item.id} className={`nav-item ${view === item.id ? "active" : ""}`} aria-current={view === item.id ? "page" : undefined} onClick={() => setView(item.id)}>
              <Icon name={item.icon} /><span>{item.label}</span>{view === item.id && <span className="nav-marker" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className={`readiness-badge ${readinessTone}`}>
            <StatusDot tone={readinessTone} />
            <div><span className="meta-label">System status</span><strong>{readinessLabel}</strong></div>
          </div>
          <p className="side-note">Built for careful decisions.<br />No diagnosis is confirmed here.</p>
          <span className="version">VERDANT / 0.1.0</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><span className="breadcrumb">Field laboratory <b>/</b> {view}</span><p className="mobile-title">Verdant</p></div>
          <div className="top-status"><StatusDot tone={readinessTone} /><span>{health ? "API connected" : "Connecting to API"}</span></div>
        </header>

        <div className="content-wrap">
          <div className="page-intro">
            <div><p className="eyebrow">Plant health instrument · 01</p><h1>{view === "overview" ? <>Read the leaf.<br /><em>Respect the uncertainty.</em></> : view === "analysis" ? <>A new <em>field note.</em></> : <>Analysis <em>history.</em></>}</h1><p className="intro-copy">{subtitle}</p></div>
            {view !== "analysis" && <button className="primary-button" onClick={() => setView("analysis")}>Start an analysis <Icon name="arrow" /></button>}
          </div>

          {view === "overview" && (
            <>
              <section className="hero-grid">
                <div className="field-note">
                  <div className="note-top"><span>FIELD NOTE / 001</span><span>{new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }).toUpperCase()}</span></div>
                  <figure className="specimen-illustration">
                    <svg viewBox="0 0 220 250" role="img" aria-labelledby="specimen-title specimen-description">
                      <title id="specimen-title">Illustrative leaf specimen</title>
                      <desc id="specimen-description">A decorative green leaf with drawn veins and warm-colored markings. It is not an analysis input or model result.</desc>
                      <defs>
                        <linearGradient id="specimen-leaf" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#e0e96b" />
                          <stop offset=".48" stopColor="#9cc94b" />
                          <stop offset="1" stopColor="#3f985f" />
                        </linearGradient>
                        <linearGradient id="specimen-shadow" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#f2d579" />
                          <stop offset="1" stopColor="#d9814d" />
                        </linearGradient>
                      </defs>
                      <path d="M35 206C15 150 28 82 96 30c24-18 54-22 89-18-1 48-13 91-39 127-30 41-68 64-111 67Z" fill="url(#specimen-leaf)" stroke="#d7e69a" strokeWidth="2" />
                      <path d="M34 207c41-52 82-101 149-192M62 174c-6-28-15-47-31-59m58 23c-3-31-1-56 8-83m-10 110c25-25 46-37 75-43m-99 72c-2-26 3-46 14-66m23 44c20-12 37-18 57-19" fill="none" stroke="#f1f0be" strokeLinecap="round" strokeWidth="2.4" opacity=".8" />
                      <path d="M51 128c10-4 18 2 17 10-1 7-8 12-17 9-7-3-8-14 0-19Zm39-46c8-5 17 0 17 8 0 9-9 13-17 9-7-4-7-13 0-17Zm42 53c7-4 15 0 15 7 0 8-9 12-16 7-6-4-5-11 1-14Zm11-71c6-4 14 0 14 6 0 7-8 10-14 6-5-3-5-9 0-12Z" fill="url(#specimen-shadow)" opacity=".92" />
                      <path d="M39 202c38-50 80-98 143-181" fill="none" stroke="#244b34" strokeLinecap="round" strokeWidth="1.5" opacity=".65" />
                    </svg>
                    <figcaption>Illustrative specimen<br />Not an analysis result</figcaption>
                  </figure>
                  <div className="specimen-line"><span className="specimen-dot" /><span>LIVE MODEL READINESS</span></div>
                  <strong>{readinessLabel}</strong>
                  <p>{health?.message || "Connect the FastAPI service to inspect checkpoint readiness."}</p>
                  <div className="note-stamp"><span>STATUS</span><b>{health?.status?.toUpperCase() || "—"}</b></div>
                </div>
                <div className="workflow">
                  <div className="section-heading"><span className="meta-label">The workflow</span><span className="section-index">03 steps</span></div>
                  {["Upload a clear leaf image", "Check input quality", "Review measured output"].map((step, index) => <div className="workflow-step" key={step}><span>0{index + 1}</span><p>{step}</p><Icon name="arrow" /></div>)}
                </div>
              </section>
              <section className="section-block">
                <div className="section-heading"><span className="meta-label">Instrument readiness</span><span className="section-index">Live from backend /health</span></div>
                <div className="readiness-grid">
                  <ReadinessCard label="Classifier checkpoint" value={classifierReady ? "Online" : "Not ready"} detail={classifierReady ? "Ready to classify" : "Trained checkpoint required"} ready={classifierReady} icon="leaf" />
                  <ReadinessCard label="Segmentation checkpoint" value={segmentationReady ? "Online" : "Not ready"} detail={segmentationReady ? "Visible area available" : "Checkpoint not configured"} ready={segmentationReady} icon="grid" />
                  <ReadinessCard label="Runtime" value="CPU mode" detail={health ? "Local inference path" : "Awaiting health response"} ready={Boolean(health)} icon="history" />
                </div>
              </section>
              <section className="notice-strip"><span className="notice-symbol">!</span><p><strong>Referral policy.</strong> Unsupported, low-quality, or uncertain images should be referred for human review. Automated output is an aid — not a confirmed diagnosis.</p></section>
              {error && <div className="error-banner" role="alert">Could not reach the backend: {error}. Start FastAPI on port 8000 to see live readiness.</div>}
            </>
          )}

          {view === "analysis" && (
            <section className="analysis-layout">
              <div className="upload-panel">
                <div className="section-heading"><span className="meta-label">Input specimen</span><span className="section-index">JPEG · PNG · WEBP</span></div>
                <div className="mode-switch" role="group" aria-label="Analysis mode">
                  <button className={analysisMode === "leaf" ? "active" : ""} aria-pressed={analysisMode === "leaf"} onClick={() => { setAnalysisMode("leaf"); setAnalysisResult(null); }}>Single leaf</button>
                  <button className={analysisMode === "whole" ? "active" : ""} aria-pressed={analysisMode === "whole"} onClick={() => { setAnalysisMode("whole"); setAnalysisResult(null); }}>Whole plant</button>
                </div>
                <div className="camera-controls">
                  {!cameraActive ? (
                    <button className="history-open-button" type="button" disabled={cameraStarting} onClick={startCamera}>
                      {cameraStarting ? "Requesting camera access…" : "Use webcam to capture"}
                    </button>
                  ) : (
                    <>
                      <video ref={cameraVideo} className="camera-preview" autoPlay playsInline muted aria-label="Live plant camera preview" />
                      <div className="camera-action-row">
                        <button className="primary-button" type="button" onClick={captureCameraImage}>Capture photo</button>
                        <button className="history-open-button" type="button" onClick={() => { stopCamera(); setCameraError(""); }}>Turn camera off</button>
                      </div>
                      <p className="camera-help">After capture, review the photo below and select Screen image to analyze it.</p>
                    </>
                  )}
                  {cameraError && <div className="warning-banner" role="status">{cameraError}</div>}
                </div>
                <div
                  className={`dropzone ${file ? "has-file" : ""}`}
                  role="button"
                  tabIndex={0}
                  aria-label="Choose a plant image or drop one here"
                  onClick={() => fileInput.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      fileInput.current?.click();
                    }
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => { event.preventDefault(); handleFile(event.dataTransfer.files); }}
                >
                  {preview ? <img src={preview} alt="Selected plant specimen" /> : <><div className="upload-icon"><Icon name="upload" /></div><strong>Drop a leaf or plant image here</strong><p>or choose a file from your device</p></>}
                  <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onClick={(event) => event.stopPropagation()} onChange={(event) => { handleFile(event.target.files); event.target.value = ""; }} />
                </div>
                {file && <div className="file-row"><span><StatusDot tone="good" />{file.name}</span><button onClick={(event) => { event.stopPropagation(); setFile(null); setPreview(""); }}>Remove</button></div>}
                <div className="validation-copy"><span className="meta-label">{historyRestoreId ? "Restore saved history" : "Before you submit"}</span><p>{historyRestoreId ? "Choose the original image used for this older history entry, then press Screen image. This browser does not have the original upload or full result, so reanalysis is required." : analysisMode === "whole" ? "Use a sharp photo that shows the whole plant and visible leaves. Overlapping image windows are classified, and only candidate crops are shown; they are not exact lesion masks." : "Use a sharp close-up of one leaf. The leaf is screened in overlapping crops; only non-healthy candidate crops above the score threshold are shown."}</p></div>
                <label className="retention-consent"><input type="checkbox" checked={saveHistoryConsent} onChange={(event) => setSaveHistoryConsent(event.target.checked)} /><span>Save this result and uploaded image in this browser’s History. Leave unchecked for a one-time analysis.</span></label>
                <button className="primary-button full" disabled={!file || analysisState === "loading" || analysisState === "blocked" || !classifierReady} onClick={runAnalysis}>{analysisState === "loading" ? "Screening regions…" : classifierReady ? "Screen image" : "Model unavailable"} <Icon name="arrow" /></button>
                {analysisState === "loading" && (
                  <div className="analysis-loading-box" role="status" aria-live="polite">
                    <div className="loading-spinner" aria-hidden="true" />
                    <div className="loading-text">
                      <strong>Processing specimen…</strong>
                      <p>Analyzing foliar features across multi-scale crops using MobileNetV3-Small.</p>
                    </div>
                  </div>
                )}
                {analysisState === "blocked" && <div className="warning-banner" role="status">{analysisError}</div>}
                {analysisState === "error" && <div className="error-banner" role="alert">{analysisError}</div>}
                {historyError && <div className="warning-banner" role="status">{historyError}</div>}
                {analysisState === "done" && analysisResult?.screening && <div className="screening-results">
                  <div className="results-action-bar">
                    <span className="meta-label">Specimen Analysis Output · Two-Stage Screening</span>
                    <button className="secondary-button report-button" type="button" onClick={() => window.print()}>Print / Save report as PDF</button>
                  </div>

                  {/* SECTION A: WHOLE-LEAF CLASSIFICATION */}
                  <section className="results-section section-classification" aria-labelledby="section-a-heading">
                    <div className="section-header-row">
                      <div>
                        <span className="section-tag">Section A</span>
                        <h2 id="section-a-heading" className="section-title">Whole-Leaf Classification</h2>
                        <p className="section-scope-note">Full-image classification evaluated separately from crop-level candidate screening.</p>
                      </div>
                      {analysisResult.screening.overall_prediction && (
                        <span className={`status-pill ${analysisResult.screening.overall_prediction.is_uncertain ? "pill-amber" : "pill-good"}`}>
                          <StatusDot tone={analysisResult.screening.overall_prediction.is_uncertain ? "warn" : "good"} />
                          {analysisResult.screening.overall_prediction.is_uncertain
                            ? "Uncertain"
                            : (analysisResult.screening.overall_prediction.prediction_status || "Model Prediction")}
                        </span>
                      )}
                    </div>

                    {analysisResult.screening.overall_prediction ? (
                      <div className={`classification-card ${analysisResult.screening.overall_prediction.is_uncertain ? "card-uncertain" : "card-confident"}`}>
                        <div className="card-top-row">
                          <div>
                            <span className="meta-label">
                              {analysisResult.screening.overall_prediction.is_uncertain ? "Tentative Predicted Class" : "Predicted Class"}
                            </span>
                            <div className="display-class-name">
                              {analysisResult.screening.overall_prediction.display_label || formatDisplayLabel(analysisResult.screening.overall_prediction.predicted_class)}
                            </div>
                            <span className="internal-class-id">
                              Dataset label: <code>{analysisResult.screening.overall_prediction.predicted_class}</code>
                            </span>
                          </div>
                          <div className="score-block">
                            <span className="score-value">
                              {formatScorePercent(analysisResult.screening.overall_prediction.confidence)}
                            </span>
                            <span className="score-badge">
                              {analysisResult.screening.overall_prediction.score_label || scoreLabel}
                            </span>
                          </div>
                        </div>

                        <div className={`prediction-guidance-box ${analysisResult.screening.overall_prediction.is_uncertain ? "guidance-amber" : "guidance-neutral"}`}>
                          <p className="guidance-explanation">
                            {analysisResult.screening.overall_prediction.explanation || (analysisResult.screening.overall_prediction.needs_review
                              ? "The model score is below the operational review threshold. Treat this result as tentative and seek professional agronomic confirmation."
                              : "The model score meets the operational review threshold. This is an automated screening prediction, not a certified laboratory diagnosis.")}
                          </p>
                          <p className="guidance-recommendation">
                            <strong>Recommendation: </strong>
                            {analysisResult.screening.overall_prediction.recommendation || (analysisResult.screening.overall_prediction.needs_review
                              ? "Seek confirmation from a qualified plant diagnostic clinic before taking management action."
                              : "Confirm symptoms with local agricultural extension before applying treatments.")}
                          </p>
                          {analysisResult.screening.overall_prediction.is_healthy && analysisResult.screening.overall_prediction.is_uncertain && (
                            <div className="healthy-warning-alert" role="alert">
                              <strong>Caution on healthy prediction: </strong>
                              Because this confidence score ({formatScorePercent(analysisResult.screening.overall_prediction.confidence)}) is below the operational review threshold ({Math.round((analysisResult.screening.overall_prediction.review_threshold || 0.60) * 100)}%), this specimen must <em>never</em> be assumed healthy or disease-free. A low-confidence healthy prediction is an inconclusive tentative match. Professional diagnostic inspection is strongly recommended.
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="info-box-neutral">
                        <p>Whole-leaf classification is evaluated in single-leaf mode. In whole-plant mode, candidate-region screening across multi-scale patches is reported below in Section B.</p>
                      </div>
                    )}

                    {analysisResult.quality?.is_blurry && (
                      <div className="warning-banner" role="status">
                        Specimen may be blurry (Laplacian variance score {analysisResult.quality.blur_score.toFixed(1)}). Low image sharpness reduces model reliability; consider capturing a sharper specimen photo.
                      </div>
                    )}
                  </section>

                  {/* SECTION B: CANDIDATE-REGION SCREENING */}
                  <section className="results-section section-screening" aria-labelledby="section-b-heading">
                    <div className="section-header-row">
                      <div>
                        <span className="section-tag">Section B</span>
                        <h2 id="section-b-heading" className="section-title">Candidate-Region Screening</h2>
                        <p className="section-scope-note">Separate crop-based screening for foliar regions exceeding the operational threshold.</p>
                      </div>
                      <span className={`status-pill ${analysisResult.screening.candidate_count > 0 ? "pill-info" : "pill-neutral"}`}>
                        {analysisResult.screening.screening_status || (analysisResult.screening.candidate_count ? `${analysisResult.screening.candidate_count} Candidate(s) Identified` : "No Candidate Regions Identified")}
                      </span>
                    </div>

                    <div className="screening-metrics-bar">
                      <div className="metric-item">
                        <span className="metric-label">Screening Mode</span>
                        <strong className="metric-value">{analysisResult.screening.mode === "whole_plant" ? "Whole Plant" : "Single Leaf"}</strong>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">Windows Examined</span>
                        <strong className="metric-value">{analysisResult.screening.regions_checked} crops</strong>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">Candidates Above Threshold</span>
                        <strong className="metric-value">{analysisResult.screening.candidate_count} regions</strong>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">Candidate Threshold</span>
                        <strong className="metric-value">{Math.round((analysisResult.screening.candidate_score_threshold || 0.60) * 100)}% uncalibrated score</strong>
                      </div>
                    </div>

                    <div className={`screening-status-card ${analysisResult.screening.candidate_count === 0 ? "status-no-candidates" : "status-has-candidates"}`}>
                      <p className="screening-summary-text">
                        {analysisResult.screening.screening_message || (analysisResult.screening.candidate_count
                          ? `${analysisResult.screening.candidate_count} candidate region(s) met the foliar screening threshold.`
                          : "No candidate regions crossed the screening threshold in the examined image crops.")}
                      </p>
                      <p className="screening-explanation-text">
                        {analysisResult.screening.screening_explanation || (analysisResult.screening.candidate_count === 0
                          ? "This screening result is inconclusive because no examined crop exceeded the operational screening threshold."
                          : "Each highlighted crop met the screening threshold after healthy-class filtering and non-maximum suppression.")}
                      </p>
                      <p className="screening-recommendation-text">
                        <strong>Recommendation: </strong>
                        {analysisResult.screening.screening_recommendation || (analysisResult.screening.candidate_count === 0
                          ? "Consider capturing a sharper, well-lit image showing the affected leaf area. If symptoms persist, seek confirmation from an agricultural extension service or qualified plant diagnostic expert."
                          : "Inspect the highlighted candidate crops alongside the whole-leaf assessment and seek confirmation from an agricultural extension service or qualified plant diagnostic expert before taking treatment action.")}
                      </p>
                      <div className="screening-safety-block" role="note">
                        <strong className="screening-safety-heading">Important Safety Notice</strong>
                        <p className="screening-safety-disclaimer">
                          {stripSafetyNoticePrefix(
                            analysisResult.screening.safety_disclaimer ||
                            "Lack of candidate screening regions does not establish that the leaf or plant is healthy or disease-free. Candidate screening identifies image-crop regions whose disease-class scores exceed the operational threshold. These regions are candidate areas for review, not confirmed disease or lesion locations."
                          )}
                        </p>
                      </div>
                    </div>

                    {analysisResult.screening.overlay && (
                      <figure className="overlay-figure">
                        <img src={analysisResult.screening.overlay} alt="Original image with candidate crop regions outlined in red and numbered" />
                        <figcaption>Original specimen: outlined boxes denote candidate image crops examined above threshold; they do not represent exact lesion masks or pixel segmentation.</figcaption>
                      </figure>
                    )}

                    {analysisResult.screening.explainability && (
                      <figure className="overlay-figure explainability-figure">
                        <img src={analysisResult.screening.explainability.image} alt={`Grad-CAM attribution for ${analysisResult.screening.explainability.class_name}; highlighted areas influence the class score and are not confirmed lesions`} />
                        <figcaption>Grad-CAM saliency for {analysisResult.screening.explainability.class_name}: {analysisResult.screening.explainability.explanation}</figcaption>
                      </figure>
                    )}

                    {analysisResult.screening.candidate_regions.map((region, index) => (
                      <article className="region-card" key={`${region.box.left}-${region.box.top}-${region.predicted_class}`}>
                        <img src={region.image} alt={`Candidate affected-region crop ${index + 1}; healthy-only crops are not displayed`} />
                        <div className="region-copy">
                          <span className="meta-label">Candidate region {index + 1} · x {region.box.left}–{region.box.right}, y {region.box.top}–{region.box.bottom}</span>
                          <strong className="display-class-name" style={{ fontSize: "17px", display: "block", marginTop: "2px", marginBottom: "4px" }}>
                            {region.display_label || formatDisplayLabel(region.predicted_class)}
                          </strong>
                          <span className="internal-class-id" style={{ marginBottom: "6px" }}>
                            Dataset label: <code>{region.predicted_class}</code>
                          </span>
                          <span className="confidence" style={{ marginBottom: "10px" }}>
                            {formatScorePercent(region.confidence)} {region.score_label || scoreLabel}
                          </span>
                          {region.needs_review && (
                            <p className="warning-banner" role="status" style={{ marginTop: "4px", marginBottom: "10px" }}>
                              Below the operational review threshold; confirm with a qualified plant-health professional.
                            </p>
                          )}
                          <p><b>Assessment:</b> {region.explanation}</p>
                          <p><b>Evidence:</b> {region.visible_evidence}</p>
                          <p><b>Possible cause:</b> {region.possible_causes.join(" ")}</p>
                          <p><b>Severity:</b> {region.severity}. {region.potential_impact}</p>
                          <span className="meta-label">Practical next steps</span>
                          <ul>{region.immediate_actions.map((action) => <li key={action}>{action}</li>)}</ul>
                          <p><b>Biological option:</b> {region.biological_treatment}</p>
                          <p><b>Chemical option:</b> {region.chemical_treatment}</p>
                          <span className="meta-label">General institutional references · not treatment trials</span>
                          <div className="source-list">
                            {region.sources.map((source) => (
                              <div className="source-entry" key={source.name}>
                                <a href={source.url} target="_blank" rel="noreferrer">{source.name}: {source.title}</a>
                                <span>{source.role}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </article>
                    ))}
                  </section>

                  {/* SECTION C: DISEASE UNDERSTANDING AND AGRICULTURAL KNOWLEDGE */}
                  <DiseaseInformationSection
                    diseaseInfo={activeDiseaseInfo}
                    overallPrediction={analysisResult.screening?.overall_prediction}
                    screening={analysisResult.screening}
                  />

                  {researchState !== "idle" && <section className="research-panel" aria-live="polite">
                    <div className="section-heading"><span className="meta-label">Retrieved research evidence</span><span className="section-index">Europe PMC · PubMed indexed</span></div>
                    {researchState === "loading" && <p className="research-state">Searching publication titles and abstracts for the leading crop and disease hypothesis…</p>}
                    {researchState === "none" && <p className="research-state">No non-healthy disease hypothesis was available to research.</p>}
                    {researchState === "error" && <p className="research-state warning-banner" role="status">Research lookup failed: {researchError}. The image-model assessment is unchanged and remains unconfirmed.</p>}
                    {researchState === "empty" && <p className="research-state">{researchResult?.message || "No matching publications were found."}</p>}
                    {researchResult?.results?.length > 0 && <>
                      <p className="research-state">{researchResult.message} Search terms: {researchResult.query}. Review source methods and applicability; a matching paper does not confirm this image diagnosis.</p>
                      <div className="research-list">{researchResult.results.map((source) => <article className="research-entry" key={source.doi || source.url}>
                        <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a>
                        <p>{source.authors} · {source.journal} · {source.year}{source.doi ? ` · DOI ${source.doi}` : ""}</p>
                        {source.abstract && <details><summary>View abstract</summary><p>{source.abstract}</p></details>}
                      </article>)}</div>
                    </>}
                  </section>}
                  <section className="conversation-panel" aria-label="Guided follow-up questions">
                    <div className="chat-header">
                      <div>
                        <span className="meta-label">Plant-health assistant</span>
                        <strong>Ask about this specimen</strong>
                      </div>
                      <span className={`chat-status-pill ${!chatStatus || chatStatus.ready ? "ready" : "waiting"}`}>
                        {!chatStatus
                          ? "Checking"
                          : chatStatus.configured
                            ? (chatStatus.ready ? "Online" : "Unavailable")
                            : "English · Ready"}
                      </span>
                    </div>
                    <p className="conversation-disclosure">
                      {chatStatus?.configured
                        ? "Your question, recent chat turns, assistant notes, and specimen analysis summary are used to answer. Uploaded images are never sent."
                        : "The assistant currently supports English and answers using this specimen's analysis results. Uploaded images are never sent to external chat services."}
                    </p>
                    <div className="agent-context-strip">
                      <span>Context memory: {agentMemoryLength} turns</span>
                      <span>Assistant notes: {assistantNotes.trim() ? "set" : "none"}</span>
                      <span>Language: {chatStatus?.configured ? "Multilingual" : "English"}</span>
                      <span>Learning ratings: {learningStatus?.ratings || 0}</span>
                      {(learningStatus?.ratings || 0) > 0 && <button className="learning-reset-button" type="button" onClick={resetAssistantLearning} disabled={learningSaving}>Reset learned styles</button>}
                    </div>
                    <p className="learning-disclosure">Use “Remember this” on a message to keep preferences for follow-up questions.</p>
                    <details className="assistant-notes">
                      <summary>Assistant notes: preferences and field context</summary>
                      <label htmlFor="assistant-notes-input">What should it remember?</label>
                      <textarea
                        id="assistant-notes-input"
                        value={assistantNotes}
                        maxLength={1200}
                        onChange={(event) => setAssistantNotes(event.target.value)}
                        placeholder="For example: use short numbered steps; I grow tomatoes in a humid climate; define technical terms."
                        rows="3"
                      />
                      <p>Saved in this browser only. Do not enter passwords or other sensitive information.</p>
                      {assistantNotes && <button type="button" className="learning-reset-button" onClick={() => setAssistantNotes("")}>Clear notes</button>}
                    </details>
                    <div className="quick-prompts" aria-label="Suggested questions">
                      {quickPrompts.map((prompt) => (
                        <button key={prompt} type="button" className="quick-prompt" onClick={() => handleSuggestedPrompt(prompt)} disabled={chatSending}>
                          {prompt}
                        </button>
                      ))}
                    </div>
                    {chatStatus?.configured && !chatStatus.ready && (
                      <p className="warning-banner" role="status">
                        The assistant is temporarily unavailable. Please try again shortly.
                      </p>
                    )}
                    <div className="conversation-list" aria-live="polite" aria-busy={chatSending}>
                      {conversation.map((message) => <article className={`conversation-message ${message.role}`} key={message.id}>
                        <span className="meta-label">{message.role === "user" ? "You" : "Assistant"}</span>
                        <p className="conversation-message-body">{message.text || (message.isStreaming ? "Generating response…" : "")}</p>
                        {message.role === "user" && <button className="remember-message-button" type="button" disabled={message.remembered} onClick={() => rememberUserMessage(message.id)}>{message.remembered ? "Remembered" : "Remember this"}</button>}
                        {message.citations?.length > 0 && <div className="conversation-citations">{message.citations.map((citation) => <a key={citation.url} href={citation.url} target="_blank" rel="noreferrer">{citation.title}</a>)}</div>}
                        {message.role === "assistant" && message.learningId && !message.isStreaming && <div className="answer-feedback" aria-label="Rate this response">
                          <span>{message.feedbackSubmitting ? "Saving rating…" : message.learningFeedback != null ? "Rating saved" : "Was this helpful?"}</span>
                          <button type="button" aria-pressed={message.learningFeedback === true} className={message.learningFeedback === true ? "selected" : ""} disabled={message.feedbackSubmitting || message.learningFeedback != null} onClick={() => rateAssistantAnswer(message.id, true)}>Helpful</button>
                          <button type="button" aria-pressed={message.learningFeedback === false} className={message.learningFeedback === false ? "selected" : ""} disabled={message.feedbackSubmitting || message.learningFeedback != null} onClick={() => rateAssistantAnswer(message.id, false)}>Not helpful</button>
                        </div>}
                        {message.role === "assistant" && !message.isStreaming && message.text && <button className="speech-button" type="button" title="Read response aloud" aria-label="Read response aloud" onClick={() => speakText(message.text)}><Icon name="speaker" /></button>}
                      </article>)}
                    </div>
                    <form className="conversation-form" onSubmit={submitChatQuestion}>
                      <label className="sr-only" htmlFor="plant-question">Ask about your specimen or model results</label>
                      <textarea
                        id="plant-question"
                        value={chatQuestion}
                        maxLength={1000}
                        disabled={chatSending}
                        onChange={(event) => setChatQuestion(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            if (chatQuestion.trim() && !chatSending && !chatSendingRef.current) {
                              submitChatQuestion(event);
                            }
                          }
                        }}
                        placeholder="Ask about your specimen or model results..."
                        rows="2"
                      />
                      <div className="conversation-actions">
                        <button
                          className="history-open-button"
                          type="button"
                          onClick={startVoiceInput}
                          disabled={!voiceInputSupported || chatSending}
                          title={voiceInputSupported ? (voiceListening ? "Stop voice input" : "Speak your question") : "Voice input is not currently supported in this browser"}
                        >
                          <Icon name="mic" />
                          {!voiceInputSupported
                            ? "Voice input unavailable"
                            : (voiceListening ? "Listening… (click to stop)" : "Speak question")}
                        </button>
                        {speechPlaying && <button className="history-open-button" type="button" onClick={stopSpeech}><Icon name="stop" />Stop audio</button>}
                        <button
                          className="chat-send-button"
                          type="submit"
                          disabled={!chatQuestion.trim() || chatSending}
                          title={chatSending ? "Generating response…" : "Send question"}
                          aria-label={chatSending ? "Generating response…" : "Send question"}
                        >
                          {chatSending ? <span className="chat-send-spinner" aria-hidden="true" /> : <Icon name="arrow" />}
                        </button>
                      </div>
                      {!voiceInputSupported && (
                        <p className="voice-unavailable-note">Voice input is not currently supported in this browser. Type your question above.</p>
                      )}
                    </form>
                    {chatError && <p className="warning-banner" role="status">{chatError}</p>}
                    <div ref={conversationEndRef} />
                  </section>
                  <div className="screening-limitations">{analysisResult.screening.limitations.map((limitation) => <p key={limitation}>{limitation}</p>)}<p>{analysisResult.screening.professional_attribution}</p><p><b>Professional sources:</b> {analysisResult.screening.sources.map((source) => source.name).join("; ")}.</p></div>
                  <article className="print-report" aria-label="Printable plant screening report">
                    <header className="print-report-header"><p>VERDANT · PLANT HEALTH SCREENING</p><h1>Image screening report</h1><p>Generated: {new Date().toLocaleString()} · File: {file?.name || "Uploaded image"}</p><p>Screening mode: {analysisResult.screening.mode === "whole_plant" ? "Whole plant" : "Single leaf"}</p></header>
                    <section>
                      <h2>Section A: Whole-Leaf Classification</h2>
                      {analysisResult.screening.overall_prediction ? (
                        <div>
                          <p><b>Predicted Class:</b> {analysisResult.screening.overall_prediction.display_label || formatDisplayLabel(analysisResult.screening.overall_prediction.predicted_class)} (<code>{analysisResult.screening.overall_prediction.predicted_class}</code>)</p>
                          <p><b>Model Score:</b> {formatScorePercent(analysisResult.screening.overall_prediction.confidence)} ({analysisResult.screening.overall_prediction.score_label || scoreLabel})</p>
                          <p><b>Status:</b> {analysisResult.screening.overall_prediction.prediction_status || (analysisResult.screening.overall_prediction.needs_review ? "Uncertain Prediction" : "Model Prediction")}</p>
                          <p><b>Explanation:</b> {analysisResult.screening.overall_prediction.explanation}</p>
                          <p><b>Recommendation:</b> {analysisResult.screening.overall_prediction.recommendation}</p>
                        </div>
                      ) : (
                        <p>Whole-leaf classification was not evaluated in whole-plant mode.</p>
                      )}
                    </section>
                    <section>
                      <h2>Section B: Candidate-Region Screening</h2>
                      <p><b>Screening Status:</b> {analysisResult.screening.screening_status || (analysisResult.screening.candidate_count ? `${analysisResult.screening.candidate_count} candidates identified` : "No Candidate Regions Identified")}</p>
                      <p>{analysisResult.screening.candidate_count} candidate region(s) identified from {analysisResult.screening.regions_checked} scanned crops at {Math.round((analysisResult.screening.candidate_score_threshold || 0.60) * 100)}% uncalibrated score threshold.</p>
                      {analysisResult.screening.screening_explanation && <p>{analysisResult.screening.screening_explanation}</p>}
                      {analysisResult.screening.screening_recommendation && <p><b>Recommendation:</b> {analysisResult.screening.screening_recommendation}</p>}
                      <p><b>Important Safety Notice:</b> {stripSafetyNoticePrefix(analysisResult.screening.safety_disclaimer || "Lack of candidate screening regions does not establish that the leaf or plant is healthy or disease-free.")}</p>
                    </section>
                    <section><h2>Original image with candidate crop locations</h2><img className="print-main-image" src={analysisResult.screening.overlay || preview} alt="Original uploaded image with candidate crop locations marked" /><p>Outlined boxes show candidate image crops, not exact diseased pixels or verified lesion boundaries.</p></section>
                    {analysisResult.screening.explainability && <section><h2>Grad-CAM model attribution</h2><img className="print-main-image" src={analysisResult.screening.explainability.image} alt={`Grad-CAM attribution for ${analysisResult.screening.explainability.class_name}`} /><p>{analysisResult.screening.explainability.explanation}</p></section>}
                    <section><h2>Candidate regions</h2>{analysisResult.screening.candidate_regions.length ? analysisResult.screening.candidate_regions.map((region, index) => <article className="print-region" key={`print-${region.box.left}-${region.box.top}-${region.predicted_class}`}><h3>Region {index + 1}: {region.display_label || formatDisplayLabel(region.predicted_class)} (<code>{region.predicted_class}</code>)</h3><img src={region.image} alt={`Candidate crop ${index + 1}`} /><p><b>Model score ({region.score_label || scoreLabel}):</b> {formatScorePercent(region.confidence)}</p>{region.needs_review && <p>Below the operational review threshold; professional confirmation is recommended.</p>}<p><b>Assessment:</b> {region.explanation}</p><p><b>Evidence:</b> {region.visible_evidence}</p><p><b>Possible cause:</b> {region.possible_causes.join(" ")}</p><p><b>Severity/impact:</b> {region.severity}; {region.potential_impact}</p><p><b>Suggested safe actions:</b></p><ul>{region.immediate_actions.map((action) => <li key={`print-${action}`}>{action}</li>)}</ul><p><b>Biological treatment:</b> {region.biological_treatment}</p><p><b>Chemical treatment:</b> {region.chemical_treatment}</p></article>) : <p>No crop met the candidate threshold. This does not demonstrate that the plant is healthy.</p>}</section>
                    {activeDiseaseInfo && (
                      <section className="print-disease-info">
                        <h2>Section C: {activeDiseaseInfo.title || "Disease Information"}</h2>
                        <p className="print-disclaimer"><b>Educational Disclaimer:</b> {activeDiseaseInfo.disclaimer}</p>
                        {activeDiseaseInfo.overview && (
                          <div>
                            <h3>Overview</h3>
                            <p><b>Common Name:</b> {activeDiseaseInfo.overview.common_name} {activeDiseaseInfo.overview.scientific_name ? `(${activeDiseaseInfo.overview.scientific_name})` : ""}</p>
                            <p><b>Pathogen Type:</b> {activeDiseaseInfo.overview.pathogen_type}</p>
                            <p>{activeDiseaseInfo.overview.description}</p>
                          </div>
                        )}
                        {activeDiseaseInfo.symptoms && (
                          <div>
                            <h3>Symptoms</h3>
                            <p><b>Leaf Symptoms:</b> {activeDiseaseInfo.symptoms.leaf_symptoms}</p>
                            <p><b>Lesion Appearance:</b> {activeDiseaseInfo.symptoms.lesion_appearance}</p>
                            <p><b>Progression:</b> {activeDiseaseInfo.symptoms.progression}</p>
                            <p><b>Other Plant Organs:</b> {activeDiseaseInfo.symptoms.other_parts_affected}</p>
                          </div>
                        )}
                        {activeDiseaseInfo.causes_and_spread && (
                          <div>
                            <h3>Causes &amp; Epidemiological Spread</h3>
                            <p><b>Causal Organism:</b> <i>{activeDiseaseInfo.causes_and_spread.causal_organism}</i></p>
                            <p><b>Favorable Environmental Conditions:</b> {activeDiseaseInfo.causes_and_spread.favorable_conditions}</p>
                            <p><b>Transmission Modes:</b> {activeDiseaseInfo.causes_and_spread.transmission_modes}</p>
                            <p><b>Host Susceptibility:</b> {activeDiseaseInfo.causes_and_spread.susceptibility_factors}</p>
                          </div>
                        )}
                        {activeDiseaseInfo.management && (
                          <div>
                            <h3>Possible Management Practices</h3>
                            <p><b>Cultural Practices:</b> {activeDiseaseInfo.management.cultural_practices}</p>
                            <p><b>Sanitation:</b> {activeDiseaseInfo.management.sanitation}</p>
                            <p><b>Environmental Control:</b> {activeDiseaseInfo.management.environmental_control}</p>
                            <p><b>Extension Advice:</b> {activeDiseaseInfo.management.professional_advice}</p>
                            <p><b>General Treatments:</b> {activeDiseaseInfo.management.general_treatments}</p>
                          </div>
                        )}
                        {activeDiseaseInfo.prevention && (
                          <div>
                            <h3>Preventative Hygiene &amp; Cultivars</h3>
                            <p><b>Crop Hygiene:</b> {activeDiseaseInfo.prevention.crop_hygiene}</p>
                            <p><b>Irrigation:</b> {activeDiseaseInfo.prevention.irrigation_practices}</p>
                            <p><b>Scouting:</b> {activeDiseaseInfo.prevention.monitoring_and_scouting}</p>
                            <p><b>Resistant Varieties:</b> {activeDiseaseInfo.prevention.resistant_varieties}</p>
                          </div>
                        )}
                        {activeDiseaseInfo.references && activeDiseaseInfo.references.length > 0 && (
                          <div>
                            <h3>Authoritative Extension References</h3>
                            <ul>
                              {activeDiseaseInfo.references.map((ref, idx) => (
                                <li key={`print-ref-${idx}-${ref.institution}`}>
                                  <b>{ref.institution}:</b> {ref.title} &mdash; <a href={ref.url}>{ref.url}</a>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </section>
                    )}
                    {!activeDiseaseInfo && analysisResult.screening?.candidate_count === 0 && (
                      <section className="print-disease-info">
                        <h2>Section C: Screening Outcome &mdash; No Candidate Regions Identified</h2>
                        <p className="print-disclaimer"><b>Educational Disclaimer:</b> This information is provided for educational purposes. Model predictions are not confirmed diagnoses. Disease symptoms can resemble those of other conditions. Consult a qualified agricultural expert or local agricultural extension service before making treatment decisions.</p>
                        <p>The candidate-region screening pipeline evaluated {analysisResult.screening.regions_checked || 0} candidate crops and identified 0 regions exceeding the 60% operational review threshold. Absence of localized candidates does not establish that the plant is healthy. Check undersides of leaves, monitor for symptom progression, and consult local extension if decline continues.</p>
                      </section>
                    )}
                    <section><h2>Limitations</h2><ul>{analysisResult.screening.limitations.map((limitation) => <li key={`print-${limitation}`}>{limitation}</li>)}</ul><p>{analysisResult.screening.professional_attribution}</p></section>
                    <section><h2>Institutional references</h2>{analysisResult.screening.sources.map((source) => <p key={`print-${source.name}`}><b>{source.name}.</b> <i>{source.title}.</i> {source.role} <a href={source.url}>{source.url}</a></p>)}</section>
                    {researchResult?.results?.length > 0 && <section><h2>Retrieved scholarly records</h2><p>{researchResult.message} Search terms: {researchResult.query}. These records are supporting material, not confirmation of the image diagnosis.</p>{researchResult.results.map((source) => <p key={`print-${source.doi || source.url}`}><b>{source.title}.</b> {source.authors}. <i>{source.journal}</i>, {source.year}. {source.doi && `DOI: ${source.doi}. `}<a href={source.url}>{source.url}</a>{source.abstract && <><br /><b>Abstract:</b> {source.abstract}</>}</p>)}</section>}
                    <footer>This report reproduces automated output and general references. It is not a professional diagnosis or treatment prescription.</footer>
                  </article>
                </div>}
              </div>
              <aside className="analysis-aside">
                <div className="aside-callout"><span className="notice-symbol">!</span><div><span className="meta-label">Model availability</span><strong>{modelsReady ? "Inference is available" : "Inference is not available yet"}</strong><p>{modelsReady ? "The configured checkpoints can process this image." : "This workspace has no trained checkpoints. Uploads cannot produce a prediction until setup is complete."}</p></div></div>
                <div className="aside-fact"><span className="meta-label">Supported classes</span><strong>{classText}</strong><p>{classes?.supported_plant ? `Plant: ${classes.supported_plant}` : "Loaded from the trained model mapping."}</p></div>
              </aside>
            </section>
          )}

          {view === "history" && (!historyLoaded ? (
            <section className="empty-history" aria-live="polite">
              <div className="empty-mark"><Icon name="history" /></div>
              <p className="eyebrow">Loading history</p>
              <h2>Restoring your saved field notes.</h2>
            </section>
          ) : history.length ? (
            <section className="history-list">
              <div className="section-heading">
                <span className="meta-label">Completed analyses · {history.length}</span>
                <div className="history-actions">
                  <button className="history-open-button" disabled={exportingHistory} onClick={exportHistory}>
                    {exportingHistory ? "Preparing export…" : "Export history"}
                  </button>
                  <button className="history-open-button" disabled={clearingHistory || Boolean(deletingHistoryId)} onClick={removeAllHistory}>
                    {clearingHistory ? "Deleting history…" : "Delete all history"}
                  </button>
                </div>
              </div>
              <p className="intro-copy">Saved in this browser and available to reopen later.</p>
              {historyError && <div className="warning-banner" role="status">{historyError}</div>}
              {history.map((record) => (
                <article className="history-row" key={record.id}>
                  <div>
                    <span className="meta-label">{record.createdAt ? new Date(record.createdAt).toLocaleString() : "Date not recorded"}</span>
                    <strong>{record.predictedClass || "Saved analysis"}</strong>
                    <p>{record.filename || "Image filename not recorded"}</p>
                    <button className="history-open-button" disabled={openingHistoryId === record.id || Boolean(deletingHistoryId) || clearingHistory} onClick={() => reopenHistoryRecord(record)}>
                      {openingHistoryId === record.id ? "Opening saved analysis…" : "Open saved analysis"}
                    </button>
                    {historyRestoreId === record.id && <button className="history-open-button history-restore-button" onClick={() => beginHistoryRestore(record.id)}>Select original image and restore</button>}
                    <button className="history-open-button" disabled={Boolean(deletingHistoryId) || clearingHistory} onClick={() => removeHistoryRecord(record)}>
                      {deletingHistoryId === record.id ? "Deleting…" : "Delete this analysis"}
                    </button>
                  </div>
                  <span className="confidence">{typeof record.confidence === "number" ? `${formatScorePercent(record.confidence)} score` : "Score not reported"}</span>
                </article>
              ))}
              <button className="secondary-button" onClick={() => setView("analysis")}>Start a new analysis <Icon name="arrow" /></button>
            </section>
          ) : (
            <section className="empty-history">
              <div className="empty-mark"><Icon name="history" /></div>
              <p className="eyebrow">No records yet</p>
              <h2>Your field notes will live here.</h2>
              <p>Every completed analysis is saved in this browser so you can reopen or delete it later.</p>
              {historyError && <div className="warning-banner" role="status">{historyError}</div>}
              <button className="secondary-button" onClick={() => setView("analysis")}>Start a new analysis <Icon name="arrow" /></button>
            </section>
          ))}
        </div>
        <footer className="footer-note">Automated image-based assessment · Not a confirmed diagnosis · Refer uncertain cases to a qualified reviewer</footer>
      </main>
    </div>
  );
}
