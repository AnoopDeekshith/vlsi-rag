import { useState, useEffect, useRef } from "react";
import {
  Upload, MessageSquare, FileText, Settings, Trash2, ChevronRight,
  Cpu, Zap, BookOpen, Loader2, Send, GraduationCap, Image, Menu, X,
  DollarSign, Download
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

/* ══════════════════════════════════════════════════════════
   RETRIEVAL ENGINE — BM25 full-text search over chunks
   ══════════════════════════════════════════════════════════ */

const STOP_WORDS = new Set([
  "a","an","the","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","shall","should","may","might","can","could",
  "in","on","at","to","for","of","with","by","from","as","into","about","between",
  "through","after","before","during","this","that","these","those","it","its",
  "and","or","but","not","no","nor","so","if","then","than","when","where","which",
  "what","who","whom","how","all","each","every","both","few","more","most","other",
  "some","such","only","own","same","just","also","very","often","still"
]);

const tokenize = (text) =>
  text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t));

const stem = (word) => {
  // Minimal suffix stripping for better recall
  return word
    .replace(/ing$/, "").replace(/tion$/, "t").replace(/sion$/, "s")
    .replace(/ment$/, "").replace(/ness$/, "").replace(/able$/, "")
    .replace(/ible$/, "").replace(/ous$/, "").replace(/ive$/, "")
    .replace(/ful$/, "").replace(/less$/, "").replace(/ly$/, "")
    .replace(/es$/, "").replace(/s$/, "").replace(/ed$/, "")
    || word;
};

const tokenizeWithStems = (text) => {
  const raw = tokenize(text);
  const expanded = new Set();
  raw.forEach((t) => { expanded.add(t); expanded.add(stem(t)); });
  return [...expanded];
};

const chunkText = (text, source, maxWords = 300, overlap = 50) => {
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim().length > 15);
  const chunks = [];
  let buffer = "";
  let prevTail = "";

  for (const para of paragraphs) {
    const words = para.split(/\s+/);
    if (buffer.split(/\s+/).length + words.length > maxWords && buffer.trim()) {
      chunks.push({
        text: (prevTail + " " + buffer).trim(),
        source,
        id: `${source}-${chunks.length}`,
      });
      const bufWords = buffer.split(/\s+/);
      prevTail = bufWords.slice(-overlap).join(" ");
      buffer = "";
    }
    buffer += "\n\n" + para;
  }
  if (buffer.trim()) {
    chunks.push({
      text: (prevTail + " " + buffer).trim(),
      source,
      id: `${source}-${chunks.length}`,
    });
  }
  return chunks.length ? chunks : [{ text: text.trim(), source, id: `${source}-0` }];
};

const buildIndex = (chunks) => {
  const N = chunks.length;
  const df = {};
  const tfs = [];
  const avgDl = chunks.reduce((s, c) => s + tokenizeWithStems(c.text).length, 0) / (N || 1);

  chunks.forEach((chunk) => {
    const tokens = tokenizeWithStems(chunk.text);
    const tf = {};
    tokens.forEach((t) => (tf[t] = (tf[t] || 0) + 1));
    tfs.push({ tf, len: tokens.length });
    Object.keys(tf).forEach((t) => (df[t] = (df[t] || 0) + 1));
  });

  return { N, df, tfs, avgDl };
};

const bm25Search = (query, chunks, index, topK = 6) => {
  const qTokens = tokenizeWithStems(query);
  const { N, df, tfs, avgDl } = index;
  const k1 = 1.5, b = 0.75;

  const scores = chunks.map((chunk, i) => {
    const { tf, len } = tfs[i];
    let score = 0;
    qTokens.forEach((qt) => {
      if (!tf[qt]) return;
      const idf = Math.log((N - (df[qt] || 0) + 0.5) / ((df[qt] || 0) + 0.5) + 1);
      score += idf * ((tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + (b * len) / avgDl)));
    });
    return { chunk, score };
  });

  return scores
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .filter((s) => s.score > 0);
};

/* ══════════════════════════════════════════════════════════
   LOCAL STORAGE HELPERS
   ══════════════════════════════════════════════════════════ */

const LS_KEYS = { apiKey: "vlsi-rag-key", docs: "vlsi-rag-docs", chat: "vlsi-rag-chat" };

const lsGet = (key) => {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : null; }
  catch { return null; }
};
const lsSet = (key, val) => {
  try { localStorage.setItem(key, JSON.stringify(val)); }
  catch (e) { console.warn("localStorage full or unavailable:", e); }
};

/* ══════════════════════════════════════════════════════════
   CLAUDE API INTERFACE
   ══════════════════════════════════════════════════════════ */

const SYSTEM_PROMPT = `You are a VLSI Design course assistant. You specialize in:
- MOS device physics: threshold voltage, I-V characteristics, band diagrams, MOS capacitors, carrier transport
- CMOS circuit design: logic gates, sizing (logical effort), power dissipation (dynamic, static, leakage), delay optimization
- Static Timing Analysis: setup/hold time, slack, clock skew/jitter, arrival/required times, multi-corner analysis
- Physical design: synthesis, floorplanning, placement, CTS, routing, DRC/LVS, IR drop, electromigration
- Verilog/SystemVerilog: RTL design, FSMs, testbenches, synthesis constraints
- Semiconductor physics: pn junctions, band diagrams, carrier concentrations, depletion regions

When answering:
1. Reference the provided course material context when available, citing the source filename
2. Use precise technical terminology
3. Write equations in LaTeX (e.g., $I_D = \\frac{\\mu_n C_{ox}}{2} \\frac{W}{L}(V_{GS}-V_T)^2$)
4. For numerical problems, show step-by-step working
5. For exam prep, highlight key concepts, common mistakes, and edge cases
6. When discussing Verilog, explain both behavior and synthesis implications
7. If context doesn't contain relevant info, say so and answer from general knowledge

Format responses in Markdown with clear section headers when appropriate.`;

const callClaudeAPIStreaming = async (apiKey, userMsg, context, onChunk) => {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      stream: true,
      system: SYSTEM_PROMPT + "\n\n" + context,
      messages: [{ role: "user", content: userMsg }],
    }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error?.message || `API error ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === "content_block_delta" && parsed.delta?.text) {
          fullText += parsed.delta.text;
          onChunk(fullText);
        }
      } catch {}
    }
  }

  return fullText || "No response received.";
};

// Non-streaming version for PDF/image extraction and exam prep
const callClaudeAPI = async (apiKey, userMsg, context) => {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      system: SYSTEM_PROMPT + "\n\n" + context,
      messages: [{ role: "user", content: userMsg }],
    }),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error.message);
  return data.content?.map((c) => c.text || "").join("\n") || "No response received.";
};

const extractPDF = async (apiKey, b64Data) => {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 8192,
      messages: [{
        role: "user",
        content: [
          { type: "document", source: { type: "base64", media_type: "application/pdf", data: b64Data } },
          { type: "text", text: `Extract ALL content from this VLSI/semiconductor document thoroughly. For each page:

TEXT: Extract all text, equations (write in LaTeX like $V_T$, $I_D$), code snippets, table data, and figure captions. Preserve section headers and structure.

IMAGES/DIAGRAMS (CRITICAL — describe every one in detail):
- Circuit schematics: component types, connections, node labels, W/L ratios, voltage/current values
- Band diagrams: energy levels (Ec, Ev, Ef, Ei), material regions, depletion widths, band bending direction
- Layout/floorplan: block names, dimensions, placement, routing layers, metal stack
- Timing diagrams: signal names, transitions, setup/hold windows, clock edges, slack values
- I-V/C-V plots: axis labels, curve shapes, key operating points, regions marked
- Cross-sections: layer stack, doping regions, oxide thickness, channel length
- Block diagrams: module names, data flow, bus widths, control signals
- Truth tables/K-maps/state diagrams: all entries and transitions

Wrap each image description in [IMAGE: ...]. Include ALL visible text labels, numerical values, and annotations. Output only extracted content.` }
        ]
      }]
    }),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error.message);
  return data.content?.map((c) => c.text || "").join("\n") || "";
};

const extractImage = async (apiKey, b64Data, mimeType, fileName) => {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: mimeType, data: b64Data } },
          { type: "text", text: `Describe this VLSI/semiconductor image in exhaustive detail for a searchable knowledge base. Identify:
- Circuit schematics: every component, connection, node label, W/L ratio, voltage/current annotation
- Band diagrams: energy levels (Ec, Ev, Ef, Ei), regions, depletion widths, band bending
- Layout/floorplan: block names, dimensions, metal layers, routing
- Timing diagrams: signal names, edges, setup/hold windows, periods, slack
- I-V or C-V plots: axes, curve shapes, operating points, regions (linear, saturation, subthreshold)
- Cross-sections: layers, doping, oxide thickness, channel dimensions
- Block/datapath diagrams: modules, data flow, control signals, bus widths
- Equations: write in LaTeX notation
Include ALL text labels, numerical values, and annotations visible. This description is the only way this image will be found via text search.` }
        ]
      }]
    }),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error.message);
  return `[IMAGE FILE: ${fileName}]\n\n` + (data.content?.map((c) => c.text || "").join("\n") || "");
};

/* ══════════════════════════════════════════════════════════
   FILE READER UTILITY
   ══════════════════════════════════════════════════════════ */

const readFileAsBase64 = (file) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result.split(",")[1]);
    r.onerror = () => reject(new Error("File read failed"));
    r.readAsDataURL(file);
  });

const MIME_MAP = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
  webp: "image/webp", gif: "image/gif",
};

const TEXT_EXTS = [
  "txt", "md", "v", "sv", "vh", "svh", "py", "tcl",
  "sdc", "lib", "lef", "def", "spice", "sp", "cir",
  "json", "yaml", "yml", "cfg", "log", "rpt", "csv",
];

const IMAGE_EXTS = ["png", "jpg", "jpeg", "webp", "gif"];

/* ══════════════════════════════════════════════════════════
   MARKDOWN RENDERER
   ══════════════════════════════════════════════════════════ */

function MarkdownContent({ content }) {
  return (
    <div className="md-content">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   LOADING DOTS
   ══════════════════════════════════════════════════════════ */

function LoadingDots() {
  return (
    <span className="loading-dots">
      <span /><span /><span />
    </span>
  );
}

/* ══════════════════════════════════════════════════════════
   MAIN APP
   ══════════════════════════════════════════════════════════ */

export default function App() {
  const [apiKey, setApiKey] = useState(() => lsGet(LS_KEYS.apiKey) || "");
  const [docs, setDocs] = useState(() => lsGet(LS_KEYS.docs) || []);
  const [chunks, setChunks] = useState([]);
  const [index, setIndex] = useState(null);
  const [messages, setMessages] = useState(() => lsGet(LS_KEYS.chat) || []);
  const [input, setInput] = useState("");
  const [view, setView] = useState("chat");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [examTopic, setExamTopic] = useState("");
  const [examResult, setExamResult] = useState(null);
  const [examLoading, setExamLoading] = useState(false);
  const [balance, setBalance] = useState(null);
  const [balanceLoading, setBalanceLoading] = useState(false);

  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const fileRef = useRef(null);
  const textareaRef = useRef(null);

  // Rebuild index from docs on mount — try bundled repo data first, then localStorage
  useEffect(() => {
    const loadDocs = async () => {
      let loadedDocs = docs; // from localStorage init

      // Try loading pre-bundled docs from public/data/docs.json
      try {
        const resp = await fetch(import.meta.env.BASE_URL + "data/docs.json");
        if (resp.ok) {
          const bundledDocs = await resp.json();
          if (bundledDocs?.length) {
            // Merge: bundled docs + any localStorage docs not already in bundled set
            const bundledNames = new Set(bundledDocs.map((d) => d.name));
            const extraDocs = loadedDocs.filter((d) => !bundledNames.has(d.name));
            loadedDocs = [...bundledDocs, ...extraDocs];
            setDocs(loadedDocs);
            // Also persist merged set to localStorage
            lsSet(LS_KEYS.docs, loadedDocs);
          }
        }
      } catch {
        // No bundled data or fetch failed — use localStorage docs
      }

      const allChunks = loadedDocs.flatMap((d) => d.chunks || []);
      setChunks(allChunks);
      if (allChunks.length) setIndex(buildIndex(allChunks));
    };
    loadDocs();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll chat — scroll the container, not the page
  useEffect(() => {
    const container = chatContainerRef.current;
    if (container) {
      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
      });
    }
  }, [messages, loading]);

  // Persist chat
  useEffect(() => { lsSet(LS_KEYS.chat, messages); }, [messages]);

  /* ── API Key ── */
  const saveApiKey = (key) => {
    setApiKey(key);
    lsSet(LS_KEYS.apiKey, key);
  };

  /* ── Balance ── */
  const fetchBalance = async (key) => {
    if (!key) { setBalance(null); return; }
    setBalanceLoading(true);
    try {
      const resp = await fetch("https://api.anthropic.com/v1/messages/count_tokens", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      // If token counting works, the key is valid — now fetch billing
      if (resp.ok) {
        // Unfortunately the billing API isn't available via browser CORS,
        // so we track usage locally as an estimate
        setBalance("valid");
      } else {
        const err = await resp.json();
        if (err.error?.type === "authentication_error") {
          setBalance("invalid");
        } else {
          setBalance("valid");
        }
      }
    } catch {
      setBalance("unknown");
    }
    setBalanceLoading(false);
  };

  // Fetch balance when API key changes
  useEffect(() => {
    if (apiKey) fetchBalance(apiKey);
    else setBalance(null);
  }, [apiKey]);

  // Track estimated spend locally
  const [estimatedSpend, setEstimatedSpend] = useState(() => lsGet("vlsi-rag-spend") || 0);
  const addSpend = (inputTokens, outputTokens) => {
    // Sonnet pricing: $3/M input, $15/M output
    const cost = (inputTokens * 3 + outputTokens * 15) / 1_000_000;
    setEstimatedSpend((prev) => {
      const next = prev + cost;
      lsSet("vlsi-rag-spend", next);
      return next;
    });
  };

  /* ── Auto-resize textarea ── */
  const autoResize = () => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 150) + "px";
    }
  };

  /* ── File Upload ── */
  const handleFiles = async (files) => {
    if (!files.length) return;
    setUploading(true);
    const newDocs = [...docs];
    const newChunks = [...chunks];
    let processed = 0;

    for (const file of files) {
      const name = file.name;
      const ext = name.split(".").pop().toLowerCase();
      let text = "";
      processed++;
      setUploadStatus(`Processing ${processed}/${files.length}: ${name}`);

      try {
        if (TEXT_EXTS.includes(ext)) {
          text = await file.text();
        } else if (IMAGE_EXTS.includes(ext)) {
          if (apiKey) {
            const b64 = await readFileAsBase64(file);
            text = await extractImage(apiKey, b64, MIME_MAP[ext] || "image/png", name);
          } else {
            text = `[Image: ${name} — set API key in Settings to enable vision extraction]`;
          }
        } else if (ext === "pdf") {
          if (apiKey) {
            const b64 = await readFileAsBase64(file);
            text = await extractPDF(apiKey, b64);
          } else {
            text = `[PDF: ${name} — set API key in Settings to enable extraction]`;
          }
        } else {
          try { text = await file.text(); } catch { text = `[Unsupported: ${name}]`; }
        }
      } catch (e) {
        text = `[Extraction failed for ${name}: ${e.message}]`;
      }

      if (text.trim()) {
        const docChunks = chunkText(text, name);
        const doc = {
          name,
          size: file.size,
          type: ext,
          addedAt: Date.now(),
          chunkCount: docChunks.length,
          chunks: docChunks,
        };
        newDocs.push(doc);
        newChunks.push(...docChunks);
      }
    }

    setDocs(newDocs);
    setChunks(newChunks);
    setIndex(buildIndex(newChunks));
    lsSet(LS_KEYS.docs, newDocs);
    setUploading(false);
    setUploadStatus("");
  };

  const deleteDoc = (idx) => {
    const newDocs = docs.filter((_, i) => i !== idx);
    const newChunks = newDocs.flatMap((d) => d.chunks || []);
    setDocs(newDocs);
    setChunks(newChunks);
    setIndex(newChunks.length ? buildIndex(newChunks) : null);
    lsSet(LS_KEYS.docs, newDocs);
  };

  const clearAllData = () => {
    setDocs([]); setChunks([]); setIndex(null); setMessages([]);
    localStorage.removeItem(LS_KEYS.docs);
    localStorage.removeItem(LS_KEYS.chat);
  };

  /* ── Export for Repo ── */
  const exportForRepo = () => {
    if (!docs.length) return;
    const json = JSON.stringify(docs, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "docs.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ── Chat (Streaming) ── */
  const sendMessage = async () => {
    if (!input.trim() || !apiKey || loading) return;
    const userMsg = input.trim();
    setInput("");
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const results = index ? bm25Search(userMsg, chunks, index, 6) : [];
      const sources = [...new Set(results.map((r) => r.chunk.source))];
      const context = results.length
        ? "RELEVANT COURSE MATERIAL:\n\n" +
          results.map((r) =>
            `--- Source: ${r.chunk.source} (relevance: ${r.score.toFixed(2)}) ---\n${r.chunk.text}`
          ).join("\n\n")
        : "No course materials uploaded yet — answering from general VLSI knowledge.";

      // Add a placeholder assistant message that we'll update in real-time
      const placeholderIdx = messages.length + 1; // +1 for the user msg we just added
      setMessages((m) => [...m, { role: "assistant", content: "", sources, streaming: true }]);

      const finalText = await callClaudeAPIStreaming(apiKey, userMsg, context, (partialText) => {
        setMessages((m) => {
          const updated = [...m];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant") {
            updated[updated.length - 1] = { ...last, content: partialText };
          }
          return updated;
        });
      });

      // Mark streaming complete
      setMessages((m) => {
        const updated = [...m];
        const last = updated[updated.length - 1];
        if (last && last.role === "assistant") {
          updated[updated.length - 1] = { ...last, content: finalText, streaming: false };
        }
        return updated;
      });

      // Rough token estimate for spend tracking
      const inputEst = (context.length + userMsg.length) / 4;
      const outputEst = finalText.length / 4;
      addSpend(inputEst, outputEst);

    } catch (e) {
      setMessages((m) => {
        const updated = [...m];
        const last = updated[updated.length - 1];
        if (last && last.role === "assistant" && last.streaming) {
          updated[updated.length - 1] = { ...last, content: `**Error:** ${e.message}`, streaming: false };
        } else {
          updated.push({ role: "assistant", content: `**Error:** ${e.message}` });
        }
        return updated;
      });
    }
    setLoading(false);
  };

  /* ── Exam Prep ── */
  const generateExam = async () => {
    if (!apiKey || !examTopic.trim()) return;
    setExamLoading(true);
    try {
      const results = index ? bm25Search(examTopic, chunks, index, 6) : [];
      const context = results.length
        ? "COURSE MATERIAL FOR EXAM GENERATION:\n\n" +
          results.map((r) => `--- ${r.chunk.source} ---\n${r.chunk.text}`).join("\n\n")
        : "No materials available — generate from general VLSI knowledge.";

      const examSystem = `You are a VLSI exam question generator. Create a challenging but fair practice exam. Include:
- 2 conceptual/short-answer questions (5 pts each)
- 2 calculation/analysis problems with real numbers (10 pts each)
- 1 Verilog design or analysis question (10 pts)
Total: 40 points. Provide clear point breakdowns.
After all questions, provide a DETAILED answer key with full worked solutions.
Use LaTeX for all equations. Format in Markdown.`;

      const response = await callClaudeAPI(
        apiKey,
        `Generate a practice exam on: ${examTopic}. Use course material context if available.`,
        examSystem + "\n\n" + context
      );
      setExamResult(response);
      // Track spend
      const inputEst = (examSystem.length + context.length + examTopic.length) / 4;
      const outputEst = response.length / 4;
      addSpend(inputEst, outputEst);
    } catch (e) {
      setExamResult(`**Error:** ${e.message}`);
    }
    setExamLoading(false);
  };

  /* ── Helpers ── */
  const docIcon = (type) => {
    if (type === "pdf") return <FileText size={18} className="icon-red" />;
    if (["v", "sv", "vh", "svh"].includes(type)) return <Cpu size={18} className="icon-green" />;
    if (IMAGE_EXTS.includes(type)) return <Image size={18} className="icon-cyan" />;
    return <FileText size={18} className="icon-amber" />;
  };

  const formatSize = (bytes) =>
    bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

  const NAV_ITEMS = [
    { id: "chat", icon: MessageSquare, label: "Chat" },
    { id: "docs", icon: FileText, label: "Documents" },
    { id: "examprep", icon: GraduationCap, label: "Exam Prep" },
    { id: "settings", icon: Settings, label: "Settings" },
  ];

  const SUGGESTIONS = [
    "Explain MOSFET regions of operation with I-V equations",
    "How does setup time violation affect timing in STA?",
    "Derive the Elmore delay for an RC ladder",
    "Explain logical effort and parasitic delay",
    "What happens at flatband in a MOS capacitor?",
    "Compare dynamic vs leakage power in CMOS",
  ];

  const EXAM_TOPICS = [
    "MOSFET I-V Characteristics",
    "Static Timing Analysis",
    "CMOS Power Dissipation",
    "Verilog FSM Design",
    "Physical Design Flow",
    "Wire Delay & Repeaters",
    "MOS Capacitor Band Diagrams",
    "Logical Effort Sizing",
  ];

  /* ══════════════════════════════════════════════════════════
     RENDER
     ══════════════════════════════════════════════════════════ */

  return (
    <div className="app">

      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon"><Cpu size={20} /></div>
            <div className="logo-text">
              <div className="logo-title">VLSI RAG</div>
              <div className="logo-sub">Course Assistant</div>
            </div>
          </div>
          <button className="sidebar-toggle mobile-only" onClick={() => setSidebarOpen(false)}>
            <X size={18} />
          </button>
        </div>

        <nav className="nav">
          {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              className={`nav-btn ${view === id ? "active" : ""}`}
              onClick={() => { setView(id); setSidebarOpen(false); }}
            >
              <Icon size={16} /> {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-stats">
          <div className="stat"><FileText size={13} /><span>{docs.length} documents</span></div>
          <div className="stat"><Zap size={13} /><span>{chunks.length} chunks</span></div>
          <div className={`stat ${apiKey ? "connected" : "disconnected"}`}>
            <div className="status-dot" />
            <span>{apiKey ? "API Connected" : "No API Key"}</span>
          </div>
          {apiKey && (
            <div className="stat spend">
              <DollarSign size={13} />
              <span>~${estimatedSpend.toFixed(4)} spent</span>
              <button className="reset-spend" onClick={() => { setEstimatedSpend(0); lsSet("vlsi-rag-spend", 0); }} title="Reset counter">↺</button>
            </div>
          )}
        </div>
      </aside>

      {/* ── Main Panel ── */}
      <main className="main-panel">
        {/* Mobile header */}
        <div className="mobile-header">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(true)}>
            <Menu size={20} />
          </button>
          <span className="mobile-title">VLSI RAG</span>
        </div>

        {/* ━━━━ CHAT VIEW ━━━━ */}
        {view === "chat" && (
          <div className="chat-view">
            <div className="view-header">
              <div>
                <h2 className="view-title">Chat with your Notes</h2>
                <p className="view-sub">MOS physics · STA · Physical design · Verilog · Device physics</p>
              </div>
              {messages.length > 0 && (
                <button className="btn-ghost" onClick={() => setMessages([])}>Clear</button>
              )}
            </div>

            <div className="chat-messages" ref={chatContainerRef}>
              {messages.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon"><BookOpen size={32} /></div>
                  <h3>Ready to study</h3>
                  <p>Upload your lecture PDFs, images, and Verilog files in the Documents tab, then ask anything here.</p>
                  <div className="suggestions">
                    {SUGGESTIONS.map((s) => (
                      <button key={s} className="suggestion" onClick={() => setInput(s)}>{s}</button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`msg-row ${msg.role}`}>
                  <div className={`msg-bubble ${msg.role}`}>
                    {msg.role === "assistant" ? (
                      <>
                        <MarkdownContent content={msg.content} />
                        {msg.streaming && <span className="streaming-cursor">▊</span>}
                      </>
                    ) : (
                      <div className="msg-text">{msg.content}</div>
                    )}
                    {msg.sources?.length > 0 && (
                      <div className="msg-sources">
                        <span className="source-label">Sources:</span>
                        {msg.sources.map((s, j) => (
                          <span key={j} className="source-tag">{s}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && messages[messages.length - 1]?.content === "" && (
                <div className="msg-row assistant">
                  <div className="msg-bubble assistant"><LoadingDots /></div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="chat-input-area">
              <div className="input-row">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => { setInput(e.target.value); autoResize(); }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder={apiKey ? "Ask about your VLSI course materials... (Shift+Enter for new line)" : "Set your API key in Settings first"}
                  disabled={!apiKey}
                  className="chat-textarea"
                  rows={1}
                />
                <button
                  onClick={sendMessage}
                  disabled={!apiKey || !input.trim() || loading}
                  className="send-btn"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ━━━━ DOCUMENTS VIEW ━━━━ */}
        {view === "docs" && (
          <div className="scroll-view">
            <div className="view-header">
              <div>
                <h2 className="view-title">Documents</h2>
                <p className="view-sub">Upload PDFs, images, Verilog, and notes — they get chunked and indexed for retrieval</p>
              </div>
              {docs.length > 0 && (
                <button className="btn-export" onClick={exportForRepo}>
                  <Download size={14} /> Export for Repo
                </button>
              )}
            </div>

            <div
              className="drop-zone"
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
              onDrop={(e) => { e.preventDefault(); e.stopPropagation(); handleFiles(Array.from(e.dataTransfer.files)); }}
            >
              <input
                ref={fileRef} type="file" multiple style={{ display: "none" }}
                accept=".pdf,.txt,.md,.v,.sv,.vh,.svh,.py,.tcl,.sdc,.lib,.lef,.def,.spice,.sp,.cir,.png,.jpg,.jpeg,.webp,.gif,.json,.yaml,.yml,.cfg,.log,.rpt,.csv"
                onChange={(e) => handleFiles(Array.from(e.target.files))}
              />
              {uploading ? (
                <>
                  <Loader2 size={28} className="spinner icon-cyan" />
                  <p className="drop-status">{uploadStatus}</p>
                </>
              ) : (
                <>
                  <Upload size={28} className="icon-cyan" />
                  <p className="drop-title">Drop files here or click to upload</p>
                  <p className="drop-hint">PDF, images (PNG/JPG), Verilog (.v, .sv), SPICE, SDC, text, markdown, and more</p>
                </>
              )}
            </div>

            <div className="doc-list">
              {docs.length === 0 && <p className="empty-text">No documents uploaded yet</p>}
              {docs.map((doc, i) => (
                <div key={i} className="doc-card">
                  <div className="doc-icon">{docIcon(doc.type)}</div>
                  <div className="doc-info">
                    <div className="doc-name">{doc.name}</div>
                    <div className="doc-meta">{doc.chunkCount} chunks · {formatSize(doc.size)} · .{doc.type}</div>
                  </div>
                  <button className="delete-btn" onClick={() => deleteDoc(i)}><Trash2 size={14} /></button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ━━━━ EXAM PREP VIEW ━━━━ */}
        {view === "examprep" && (
          <div className="scroll-view">
            <div className="view-header">
              <div>
                <h2 className="view-title">Exam Prep</h2>
                <p className="view-sub">Generate practice exams with worked solutions from your course materials</p>
              </div>
            </div>

            <div className="exam-input-section">
              <p className="exam-hint">Enter a topic and get a 40-point practice exam with a detailed answer key. Questions are generated from your uploaded materials when available.</p>
              <div className="input-row">
                <input
                  value={examTopic}
                  onChange={(e) => setExamTopic(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && generateExam()}
                  placeholder="e.g. MOS capacitor band diagrams, STA setup/hold, CMOS inverter sizing..."
                  disabled={!apiKey}
                  className="chat-input"
                />
                <button onClick={generateExam} disabled={!apiKey || !examTopic.trim() || examLoading} className="send-btn">
                  {examLoading ? <Loader2 size={16} className="spinner" /> : <ChevronRight size={16} />}
                </button>
              </div>
              <div className="topic-chips">
                {EXAM_TOPICS.map((t) => (
                  <button key={t} className="chip" onClick={() => setExamTopic(t)}>{t}</button>
                ))}
              </div>
            </div>

            {examResult && (
              <div className="exam-result">
                <MarkdownContent content={examResult} />
              </div>
            )}
          </div>
        )}

        {/* ━━━━ SETTINGS VIEW ━━━━ */}
        {view === "settings" && (
          <div className="scroll-view">
            <div className="view-header">
              <div>
                <h2 className="view-title">Settings</h2>
                <p className="view-sub">Configure your API key and manage data</p>
              </div>
            </div>

            <div className="settings-card">
              <h3>Anthropic API Key</h3>
              <p>Required for chat, PDF extraction, and image analysis. Your key stays in your browser's local storage and is never sent anywhere except directly to Anthropic's API.</p>
              <div className="key-steps">
                <span>1. Go to <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">console.anthropic.com/settings/keys</a></span>
                <span>2. Create a key → paste below</span>
              </div>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => saveApiKey(e.target.value)}
                placeholder="sk-ant-api03-..."
                className="chat-input mono"
              />
              <div className={`key-status ${apiKey ? "ok" : "bad"}`}>
                {apiKey ? "✓ API key set" : "✗ No API key configured"}
              </div>
            </div>

            <div className="settings-card">
              <h3>Data Management</h3>
              <p>Clear all uploaded documents, chat history, and the search index. This cannot be undone.</p>
              <button className="btn-danger" onClick={clearAllData}>
                <Trash2 size={14} /> Clear All Data
              </button>
            </div>

            <div className="settings-card">
              <h3>How It Works</h3>
              <div className="how-it-works">
                <div className="step">
                  <div className="step-num">1</div>
                  <div><strong>Upload</strong> — PDFs are sent to Claude's vision API to extract text + detailed image descriptions. Verilog/text files are read directly.</div>
                </div>
                <div className="step">
                  <div className="step-num">2</div>
                  <div><strong>Index</strong> — Content is split into ~300-word chunks with overlap. A BM25 inverted index is built in-browser for fast retrieval.</div>
                </div>
                <div className="step">
                  <div className="step-num">3</div>
                  <div><strong>Retrieve</strong> — When you ask a question, BM25 finds the top 6 most relevant chunks from your materials.</div>
                </div>
                <div className="step">
                  <div className="step-num">4</div>
                  <div><strong>Generate</strong> — The chunks are sent as context to Claude, which answers using your course materials + its VLSI knowledge.</div>
                </div>
              </div>
            </div>

            <div className="settings-card">
              <h3>Supported File Types</h3>
              <div className="file-types">
                {[".pdf", ".png", ".jpg", ".webp", ".gif", ".v", ".sv", ".vh", ".svh", ".txt", ".md", ".py", ".tcl", ".sdc", ".lib", ".lef", ".def", ".spice", ".sp", ".csv", ".log", ".rpt"].map((ext) => (
                  <span key={ext} className="file-tag">{ext}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
