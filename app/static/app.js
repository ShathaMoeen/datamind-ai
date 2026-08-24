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
    runAnalysis: "Run analysis", runningAnalysis: "Agents are analyzing your request…",
    analysisReport: "Analysis report", reportSubtitle: "Grounded findings and recommendations",
    summary: "Executive summary", findings: "Verified findings", interpretations: "Interpretations",
    recommendations: "Recommendations", limitations: "Limitations",
    uploadFirst: "Upload a dataset or document first.", questionRequired: "Enter an analytical question.",
    orchestrator: "Orchestrator", reportAgent: "Report Agent",
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
    runAnalysis: "تشغيل التحليل", runningAnalysis: "الوكلاء يحللون طلبك الآن…",
    analysisReport: "تقرير التحليل", reportSubtitle: "نتائج وتوصيات موثقة",
    summary: "الملخص التنفيذي", findings: "النتائج المؤكدة", interpretations: "التفسيرات",
    recommendations: "التوصيات", limitations: "القيود",
    uploadFirst: "ارفع مجموعة بيانات أو مستندًا أولًا.", questionRequired: "اكتب سؤالًا تحليليًا.",
    orchestrator: "المنسق", reportAgent: "وكيل التقرير",
    apiOnline: "الواجهة البرمجية متصلة", apiUnavailable: "الواجهة البرمجية غير متاحة", uploadingProfile: "جارٍ الرفع وتحليل الجودة…",
    datasetReady: "جاهز للتحليل.", uploadingDocument: "جارٍ رفع المستند…", documentReady: "جاهز للفهرسة.",
    rows: "الصفوف", columns: "الأعمدة", duplicates: "الصفوف المكررة", missingColumns: "أعمدة تحتوي قيمًا مفقودة",
  },
};

let language = localStorage.getItem("datamind-language") || "en";
let datasetId = null;
const documentIds = [];
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
    datasetId = uploaded.dataset_id;
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
    documentIds.push(uploaded.document_id);
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

const addReportSection = (container, title, items) => {
  if (!items.length) return;
  const section = document.createElement("section");
  section.className = "report-section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.appendChild(heading);
  const list = document.createElement("ul");
  items.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = item;
    list.appendChild(listItem);
  });
  section.appendChild(list);
  container.appendChild(section);
};

const renderReport = (result) => {
  const report = result.final_report.report;
  const container = document.getElementById("report-content");
  container.replaceChildren();
  const flow = document.getElementById("workflow-flow");
  flow.replaceChildren();
  const agentLabels = {
    orchestrator: t("orchestrator"), data_agent: language === "ar" ? "وكيل البيانات" : "Data Agent",
    analysis_agent: language === "ar" ? "وكيل التحليل" : "Analysis Agent",
    visualization_agent: language === "ar" ? "وكيل التصور" : "Visualization Agent",
    rag_agent: language === "ar" ? "وكيل المستندات" : "RAG Agent", report_agent: t("reportAgent"),
  };
  result.trace.forEach((event, index) => {
    if (!event.agent) return;
    if (index) {
      const arrow = document.createElement("span");
      arrow.className = "workflow-arrow";
      arrow.textContent = language === "ar" ? "←" : "→";
      flow.appendChild(arrow);
    }
    const node = document.createElement("span");
    node.className = "workflow-node";
    node.textContent = `✓ ${agentLabels[event.agent] || event.agent}`;
    flow.appendChild(node);
  });
  const kpis = document.getElementById("report-kpis");
  kpis.replaceChildren();
  result.final_report.source_facts
    .filter((fact) => typeof fact.value === "number")
    .slice(0, 4)
    .forEach((fact) => {
      const card = document.createElement("div");
      card.className = "report-kpi";
      const label = document.createElement("span");
      label.textContent = fact.statement.replace(/:\s*[\d.]+\.?$/, "");
      const value = document.createElement("strong");
      value.textContent = new Intl.NumberFormat(language).format(fact.value);
      card.append(label, value);
      kpis.appendChild(card);
    });
  const visualization = result.specialist_results.find(
    (item) => item.selected_agent === "visualization_agent"
  );
  const chartCard = document.getElementById("chart-card");
  if (visualization && window.Plotly) {
    chartCard.hidden = false;
    window.Plotly.newPlot(
      "analysis-chart",
      visualization.figure.data,
      {
        ...visualization.figure.layout,
        paper_bgcolor: "#0e100f",
        plot_bgcolor: "#0e100f",
        font: { color: "#e8e7e2" },
        margin: { t: 60, r: 30, b: 60, l: 65 },
      },
      { responsive: true, displaylogo: false }
    );
  } else {
    chartCard.hidden = true;
  }
  addReportSection(container, t("summary"), [report.executive_summary]);
  addReportSection(container, t("findings"), report.findings.map((item) => item.statement));
  addReportSection(container, t("interpretations"), report.interpretations.map((item) => item.statement));
  addReportSection(container, t("recommendations"), report.recommendations.map((item) => item.action));
  addReportSection(container, t("limitations"), report.limitations);
  document.getElementById("analysis-results").hidden = false;
};

document.getElementById("analysis-button").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const message = document.getElementById("analysis-message");
  if (!question) return showMessage(message, t("questionRequired"), true);
  if (!datasetId && !documentIds.length) return showMessage(message, t("uploadFirst"), true);
  showMessage(message, t("runningAnalysis"));
  try {
    const result = await requestJson(`${api}/analysis/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        dataset_id: datasetId,
        document_ids: documentIds,
        language,
        dashboard_mode: true,
      }),
    });
    renderReport(result);
    showMessage(message, "");
  } catch (error) {
    showMessage(message, error.message, true);
  }
});

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
