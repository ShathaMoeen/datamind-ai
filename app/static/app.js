const api = "/api/v1";

const translations = {
  en: {
    checkingApi: "Checking API", eyebrow: "AI-POWERED ANALYTICS",
    heroLine1: "Turn raw data into", heroLine2: "clear decisions.",
    heroCopy: "Upload business data and supporting documents. DataMind validates, profiles, and prepares them for grounded multi-agent analysis.",
    uploadDataset: "Upload dataset", datasetSubtitle: "CSV or Excel, securely validated",
    chooseDataset: "Choose a dataset", datasetTypes: "CSV or XLSX", uploadAndProfile: "Upload and profile",
    addEvidence: "Add evidence", evidenceSubtitle: "PDF reports for grounded answers",
    chooseDocument: "Choose a document", documentType: "Text-based PDF", uploadDocument: "Upload document",
    dataProfile: "Data profile", profileSubtitle: "Deterministic quality checks",
    profileEmpty: "Upload a dataset to see its structure and quality metrics.",
    askDataMind: "Ask DataMind", questionSubtitle: "Multi-agent analysis workspace",
    questionLabel: "Your analytical question", questionPlaceholder: "What are the most important trends in this dataset?",
    analysisDisabled: "Analysis API connection is the next step", runAnalysis: "Run analysis · coming next",
    apiOnline: "API online", apiUnavailable: "API unavailable", uploadingProfile: "Uploading and profiling…",
    datasetReady: "is ready.", uploadingDocument: "Uploading document…", documentReady: "is ready for indexing.",
    rows: "Rows", columns: "Columns", duplicates: "Duplicate rows", missingColumns: "Columns with missing data",
  },
  ar: {
    checkingApi: "جارٍ فحص الواجهة البرمجية", eyebrow: "تحليلات مدعومة بالذكاء الاصطناعي",
    heroLine1: "حوّل البيانات الخام إلى", heroLine2: "قرارات واضحة.",
    heroCopy: "ارفع بيانات العمل والمستندات الداعمة. يتحقق DataMind منها ويحلل جودتها ويجهزها لتحليل موثوق متعدد الوكلاء.",
    uploadDataset: "رفع مجموعة بيانات", datasetSubtitle: "CSV أو Excel مع تحقق آمن",
    chooseDataset: "اختر مجموعة بيانات", datasetTypes: "CSV أو XLSX", uploadAndProfile: "رفع وتحليل الجودة",
    addEvidence: "إضافة مستندات داعمة", evidenceSubtitle: "تقارير PDF لإجابات موثقة",
    chooseDocument: "اختر مستندًا", documentType: "ملف PDF نصي", uploadDocument: "رفع المستند",
    dataProfile: "ملف جودة البيانات", profileSubtitle: "فحوصات جودة حتمية",
    profileEmpty: "ارفع مجموعة بيانات لعرض بنيتها ومقاييس جودتها.",
    askDataMind: "اسأل DataMind", questionSubtitle: "مساحة تحليل متعددة الوكلاء",
    questionLabel: "سؤالك التحليلي", questionPlaceholder: "ما أهم الاتجاهات في مجموعة البيانات؟",
    analysisDisabled: "سيتم ربط واجهة التحليل البرمجية في الخطوة التالية", runAnalysis: "تشغيل التحليل · قريبًا",
    apiOnline: "الواجهة البرمجية متصلة", apiUnavailable: "الواجهة البرمجية غير متاحة", uploadingProfile: "جارٍ الرفع وتحليل الجودة…",
    datasetReady: "جاهز للتحليل.", uploadingDocument: "جارٍ رفع المستند…", documentReady: "جاهز للفهرسة.",
    rows: "الصفوف", columns: "الأعمدة", duplicates: "الصفوف المكررة", missingColumns: "أعمدة تحتوي قيمًا مفقودة",
  },
};

let language = localStorage.getItem("datamind-language") || "en";
const t = (key) => translations[language][key];

const applyLanguage = () => {
  document.documentElement.lang = language;
  document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((element) => {
    element.title = t(element.dataset.i18nTitle);
  });
  document.getElementById("language-toggle").textContent = language === "en" ? "العربية" : "English";
};

const showMessage = (element, message, isError = false) => {
  element.textContent = message;
  element.classList.toggle("error", isError);
};

const requestJson = async (url, options = {}) => {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "The request failed.");
  return body;
};

const updateFileLabel = (inputId, labelId) => {
  const input = document.getElementById(inputId);
  input.addEventListener("change", () => {
    if (input.files.length) document.getElementById(labelId).textContent = input.files[0].name;
  });
};

const renderProfile = (profile) => {
  const results = document.getElementById("profile-results");
  const metrics = [
    [t("rows"), profile.row_count],
    [t("columns"), profile.column_count],
    [t("duplicates"), profile.duplicate_rows],
    [t("missingColumns"), profile.columns.filter((item) => item.missing_count > 0).length],
  ];
  results.innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
  document.getElementById("profile-empty").hidden = true;
  results.hidden = false;
};

document.getElementById("dataset-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("dataset-file").files[0];
  const message = document.getElementById("dataset-message");
  const data = new FormData();
  data.append("file", file);
  showMessage(message, t("uploadingProfile"));
  try {
    const uploaded = await requestJson(`${api}/datasets/upload`, { method: "POST", body: data });
    const profile = await requestJson(`${api}/datasets/${uploaded.dataset_id}/profile`);
    renderProfile(profile);
    showMessage(message, `${uploaded.original_filename} ${t("datasetReady")}`);
  } catch (error) {
    showMessage(message, error.message, true);
  }
});

document.getElementById("document-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("document-file").files[0];
  const message = document.getElementById("document-message");
  const data = new FormData();
  data.append("file", file);
  showMessage(message, t("uploadingDocument"));
  try {
    const uploaded = await requestJson(`${api}/documents/upload`, { method: "POST", body: data });
    showMessage(message, `${uploaded.original_filename} ${t("documentReady")}`);
  } catch (error) {
    showMessage(message, error.message, true);
  }
});

const checkHealth = async () => {
  const status = document.getElementById("server-status");
  const statusText = document.getElementById("status-text");
  try {
    await requestJson(`${api}/health`);
    status.classList.add("online");
    statusText.textContent = t("apiOnline");
  } catch {
    status.classList.add("error");
    statusText.textContent = t("apiUnavailable");
  }
};

updateFileLabel("dataset-file", "dataset-label");
updateFileLabel("document-file", "document-label");
document.getElementById("language-toggle").addEventListener("click", () => {
  language = language === "en" ? "ar" : "en";
  localStorage.setItem("datamind-language", language);
  applyLanguage();
  checkHealth();
});
applyLanguage();
checkHealth();
