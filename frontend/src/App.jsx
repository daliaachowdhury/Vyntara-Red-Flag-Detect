import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  Upload,
  Image as ImageIcon,
  FileText,
  Sparkles,
  BarChart3,
  Cpu,
  RefreshCw,
  Copy,
  Check,
  ChevronRight,
  Flame,
  MessageSquare,
  Activity,
  Layers,
  ArrowRight,
  Globe,
  Brain,
  ThumbsUp,
  RotateCcw,
  Sliders,
  Zap
} from 'lucide-react';

const API_BASE = window.location.port === '5173' ? 'http://127.0.0.1:8000' : '';

export default function App() {
  const [activeTab, setActiveTab] = useState('text'); // 'text', 'upload', 'metrics'
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [trainingStats, setTrainingStats] = useState(null);
  const [samples, setSamples] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');
  const [copied, setCopied] = useState(false);
  const [autoLearn, setAutoLearn] = useState(true);
  
  // Feedback & Self-training state
  const [feedbackSuccess, setFeedbackSuccess] = useState('');
  const [isRetraining, setIsRetraining] = useState(false);

  // File upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchSamples();
    fetchMetrics();
    fetchTrainingStats();
  }, []);

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data);
      }
    } catch (err) {
      console.error('Error fetching samples:', err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/model/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  };

  const fetchTrainingStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/train/stats`);
      if (res.ok) {
        const data = await res.json();
        setTrainingStats(data);
      }
    } catch (err) {
      console.error('Error fetching training stats:', err);
    }
  };

  const handleAnalyzeText = async (textToAnalyze = inputText) => {
    if (!textToAnalyze.trim()) {
      setErrorMsg('Please enter or paste a conversation to analyze.');
      return;
    }
    setErrorMsg('');
    setFeedbackSuccess('');
    setIsAnalyzing(true);
    try {
      const res = await fetch(`${API_BASE}/api/analyze/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToAnalyze, auto_learn: autoLearn })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }
      const data = await res.json();
      setResult(data);
      fetchTrainingStats();
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    setErrorMsg('');
    setFeedbackSuccess('');
    setSelectedFile(file);
    
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('auto_learn', autoLearn.toString());

    try {
      const res = await fetch(`${API_BASE}/api/analyze/upload`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload analysis failed');
      }
      const data = await res.json();
      setResult(data);
      if (data.file_info?.extracted_text) {
        setInputText(data.file_info.extracted_text);
      }
      fetchTrainingStats();
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFeedback = async (assignedFlag) => {
    if (!result) return;
    setIsRetraining(true);
    setFeedbackSuccess('');
    try {
      const res = await fetch(`${API_BASE}/api/train/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText || result.file_info?.extracted_text,
          flag: assignedFlag,
          behavior: result.summary.behavior_raw || 'custom_learned',
          language: result.language_detection?.language || 'Multilingual'
        })
      });
      if (!res.ok) throw new Error('Feedback submission failed');
      const data = await res.json();
      setFeedbackSuccess(`🧠 Self-Training Complete! Model weights updated with this ${assignedFlag} pattern.`);
      fetchMetrics();
      fetchTrainingStats();
      // Re-run inference with updated model weights
      handleAnalyzeText(inputText);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsRetraining(false);
    }
  };

  const handleTriggerRetrain = async () => {
    setIsRetraining(true);
    try {
      const res = await fetch(`${API_BASE}/api/train/retrain`, { method: 'POST' });
      if (res.ok) {
        setFeedbackSuccess('Full dataset retraining complete! Model weights refreshed.');
        fetchMetrics();
        fetchTrainingStats();
      }
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsRetraining(false);
    }
  };

  const loadPreset = (sample) => {
    setInputText(sample.text);
    setImagePreview(null);
    setSelectedFile(null);
    handleAnalyzeText(sample.text);
  };

  const copyAnalysis = () => {
    if (!result) return;
    const summaryText = `🚩 REDFLAG AI Analysis
Language: ${result.language_detection?.language || 'English'}
Verdict: ${result.summary.flag} (${result.summary.confidence}% Confidence)
Behavior: ${result.summary.behavior_category}
Toxicity: ${result.sentiment_and_toxicity.toxicity_score}%

Why Flagged:
${result.explain_why.summary_reason}

Triggers: ${result.explain_why.highlighted_triggers.join(', ') || 'None'}
`;
    navigator.clipboard.writeText(summaryText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getFlagTheme = (flag) => {
    switch (flag) {
      case 'Red Flag':
        return {
          bg: 'bg-rose-500/10 border-rose-500/40 text-rose-300',
          badge: 'bg-rose-500 text-white',
          glow: 'shadow-[0_0_40px_rgba(244,63,94,0.25)]',
          icon: <ShieldAlert className="w-8 h-8 text-rose-400" />,
          color: '#f43f5e',
          textClass: 'text-rose-400'
        };
      case 'Yellow Flag':
        return {
          bg: 'bg-amber-500/10 border-amber-500/40 text-amber-300',
          badge: 'bg-amber-500 text-slate-950 font-semibold',
          glow: 'shadow-[0_0_40px_rgba(245,158,11,0.2)]',
          icon: <AlertTriangle className="w-8 h-8 text-amber-400" />,
          color: '#f59e0b',
          textClass: 'text-amber-400'
        };
      case 'Green Flag':
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300',
          badge: 'bg-emerald-500 text-slate-950 font-semibold',
          glow: 'shadow-[0_0_40px_rgba(16,185,129,0.25)]',
          icon: <ShieldCheck className="w-8 h-8 text-emerald-400" />,
          color: '#10b981',
          textClass: 'text-emerald-400'
        };
      default:
        return {
          bg: 'bg-slate-500/10 border-slate-500/40 text-slate-300',
          badge: 'bg-slate-600 text-white',
          glow: 'shadow-[0_0_30px_rgba(148,163,184,0.15)]',
          icon: <HelpCircle className="w-8 h-8 text-slate-400" />,
          color: '#94a3b8',
          textClass: 'text-slate-400'
        };
    }
  };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex flex-col selection:bg-rose-500/30">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 sticky top-0 z-50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl select-none">🚩</span>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-rose-400 via-amber-300 to-emerald-400 bg-clip-text text-transparent">
                  REDFLAG AI
                </span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300">
                  Multilingual v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                English • Hinglish • Hindi (हिन्दी) • Marathi (मराठी)
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setActiveTab('text')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'text'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <span className="flex items-center space-x-1.5">
                <MessageSquare className="w-3.5 h-3.5" />
                <span>Analyzer</span>
              </span>
            </button>

            <button
              onClick={() => setActiveTab('upload')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'upload'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <span className="flex items-center space-x-1.5">
                <Upload className="w-3.5 h-3.5" />
                <span>Screenshot OCR</span>
              </span>
            </button>

            <button
              onClick={() => setActiveTab('metrics')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'metrics'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <span className="flex items-center space-x-1.5">
                <Brain className="w-3.5 h-3.5 text-amber-400" />
                <span className="hidden md:inline">Self-Training & Metrics</span>
                <span className="md:hidden">Brain</span>
              </span>
            </button>
          </div>
        </div>
      </header>

      {/* Hero Banner with Language Indicators */}
      <section className="relative overflow-hidden py-8 px-4 sm:px-6 lg:px-8 border-b border-slate-900 bg-gradient-to-b from-slate-950 via-[#0a0f1d] to-[#080c14]">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900/80 border border-slate-800 text-xs text-slate-300 mb-3">
            <Globe className="w-3.5 h-3.5 text-rose-400" />
            <span>Multilingual Understanding: English, Hindi, Hinglish, Marathi + Adaptive Self-Training</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white mb-2">
            Multilingual Relationship Intelligence
          </h1>
          <p className="text-sm sm:text-base text-slate-400 max-w-2xl mx-auto">
            Analyze texts, Hinglish chat mix, Marathi, Hindi, or screenshots. The model continuously self-trains and adapts according to the messages you share.
          </p>

          {/* Preset Buttons for Quick Demo */}
          <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
            <span className="text-xs text-slate-400 mr-1 flex items-center">
              <Sparkles className="w-3.5 h-3.5 text-amber-400 mr-1" />
              Try multilingual scenarios:
            </span>
            {samples.map((s) => (
              <button
                key={s.id}
                onClick={() => loadPreset(s)}
                className="text-xs px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white transition-all hover:scale-[1.02]"
              >
                {s.title.split(':')[0]}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
        {activeTab === 'metrics' ? (
          /* Model Transparency & Self-Training View */
          <div className="space-y-6">
            <div className="glass-panel p-6 rounded-2xl border border-slate-800">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-xl font-bold text-white flex items-center space-x-2">
                    <Brain className="w-5 h-5 text-amber-400" />
                    <span>Active Learning & Model Brain</span>
                  </h2>
                  <p className="text-xs text-slate-400">
                    The model continuously learns from user-shared texts and screenshot OCR inputs.
                  </p>
                </div>
                <div className="flex items-center space-x-3">
                  <button
                    onClick={handleTriggerRetrain}
                    disabled={isRetraining}
                    className="px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 text-xs font-medium hover:bg-amber-500/30 flex items-center space-x-1.5 transition-all"
                  >
                    <RotateCcw className={`w-3.5 h-3.5 ${isRetraining ? 'animate-spin' : ''}`} />
                    <span>{isRetraining ? 'Retraining...' : 'Force Full Retrain'}</span>
                  </button>
                </div>
              </div>

              {/* Active Learning Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">Total Training Samples</span>
                  <p className="text-sm font-semibold text-white mt-1">
                    {metrics?.dataset_size || 1300} samples
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">User Learned Contributions</span>
                  <p className="text-sm font-semibold text-amber-300 mt-1">
                    {trainingStats?.total_learned_samples || 0} active inputs
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">Supported Languages</span>
                  <p className="text-sm font-semibold text-emerald-400 mt-1">
                    English, Hinglish, Marathi, Hindi
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-xs text-slate-400">Subword Feature Dimensions</span>
                  <p className="text-sm font-semibold text-white mt-1">
                    {metrics?.vocabulary_size || 5089} subwords
                  </p>
                </div>
              </div>

              {/* Recent Learned Samples */}
              {trainingStats?.recent_learned_samples?.length > 0 && (
                <div className="mt-8">
                  <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center space-x-2">
                    <Zap className="w-4 h-4 text-amber-400" />
                    <span>Recent Self-Trained Adaptations</span>
                  </h3>
                  <div className="space-y-2">
                    {trainingStats.recent_learned_samples.map((item, idx) => (
                      <div key={idx} className="p-3 rounded-xl bg-slate-900/40 border border-slate-800 flex items-center justify-between text-xs">
                        <span className="font-mono text-slate-300 line-clamp-1 flex-1 pr-4">"{item.text}"</span>
                        <div className="flex items-center space-x-2">
                          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">{item.language}</span>
                          <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-semibold">{item.flag}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Per-Class Report */}
              {metrics?.primary_model_evaluation?.classification_report && (
                <div className="mt-8">
                  <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-rose-400" />
                    <span>Classification Report Across Languages</span>
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-2.5 px-3">Class</th>
                          <th className="py-2.5 px-3">Precision</th>
                          <th className="py-2.5 px-3">Recall</th>
                          <th className="py-2.5 px-3">F1-Score</th>
                          <th className="py-2.5 px-3">Support</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {metrics.classes.map((c) => {
                          const row = metrics.primary_model_evaluation.classification_report[c] || {};
                          return (
                            <tr key={c} className="hover:bg-slate-900/40">
                              <td className="py-2 px-3 font-semibold text-slate-200">{c}</td>
                              <td className="py-2 px-3 text-slate-300">{(row.precision || 0).toFixed(3)}</td>
                              <td className="py-2 px-3 text-slate-300">{(row.recall || 0).toFixed(3)}</td>
                              <td className="py-2 px-3 text-emerald-400 font-semibold">{(row['f1-score'] || 0).toFixed(3)}</td>
                              <td className="py-2 px-3 text-slate-400">{row.support || 0}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Main Input & Analysis View */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column: Input Methods */}
            <div className="lg:col-span-5 space-y-4">
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
                    {activeTab === 'text' ? (
                      <>
                        <MessageSquare className="w-4 h-4 text-rose-400" />
                        <span>Paste Texts / Hinglish / Marathi</span>
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4 text-rose-400" />
                        <span>Screenshot OCR / Chat File</span>
                      </>
                    )}
                  </h2>
                  <div className="flex items-center space-x-2">
                    <label className="flex items-center space-x-1.5 cursor-pointer text-xs text-slate-400 select-none">
                      <input
                        type="checkbox"
                        checked={autoLearn}
                        onChange={(e) => setAutoLearn(e.target.checked)}
                        className="rounded border-slate-700 text-rose-500 focus:ring-rose-500 bg-slate-900"
                      />
                      <span className="text-[11px] text-amber-300 font-medium">Auto-Train</span>
                    </label>
                  </div>
                </div>

                {activeTab === 'text' ? (
                  <div>
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder={`Paste messages in ANY language...\n\nExamples:\n• Hinglish: "Agar tu mujhse pyaar karti to dosto se baat nahi karti. Live location bhej abhi."\n• Marathi: "Jar tuza mazyavar prem asel tar tu tya lokanshi bolnar nahis."\n• English: "If you leave, I'll ruin your life."`}
                      rows={9}
                      className="w-full rounded-xl bg-slate-900/90 border border-slate-800 p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-rose-500/50 focus:border-rose-500 transition-all font-mono"
                    />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Drag and Drop Zone */}
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-slate-700 hover:border-rose-500/60 rounded-xl p-6 text-center cursor-pointer transition-all bg-slate-900/40 hover:bg-slate-900/80 group"
                    >
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*,.txt,.log,.chat"
                        className="hidden"
                        onChange={(e) => {
                          if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
                        }}
                      />
                      <div className="w-12 h-12 rounded-full bg-slate-800 group-hover:bg-rose-500/20 text-slate-400 group-hover:text-rose-400 flex items-center justify-center mx-auto mb-3 transition-colors">
                        <Upload className="w-6 h-6" />
                      </div>
                      <p className="text-sm font-medium text-slate-200">
                        Drop screenshot or chat file here
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        Extracts Marathi, Hindi, Hinglish, & English text using OCR
                      </p>
                    </div>

                    {imagePreview && (
                      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                          <span className="flex items-center space-x-1">
                            <ImageIcon className="w-3.5 h-3.5 text-rose-400" />
                            <span>Uploaded Screenshot</span>
                          </span>
                          <span>{selectedFile?.name}</span>
                        </div>
                        <img
                          src={imagePreview}
                          alt="Screenshot preview"
                          className="max-h-48 w-auto mx-auto rounded-lg border border-slate-700 object-contain"
                        />
                      </div>
                    )}

                    {inputText && (
                      <div>
                        <span className="text-xs font-semibold text-slate-400 block mb-1.5">
                          Extracted Dialogue (OCR Text):
                        </span>
                        <textarea
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          rows={4}
                          className="w-full rounded-lg bg-slate-900/80 border border-slate-800 p-2.5 text-xs text-slate-200 font-mono"
                        />
                      </div>
                    )}
                  </div>
                )}

                {errorMsg && (
                  <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
                    {errorMsg}
                  </div>
                )}

                <button
                  onClick={() => handleAnalyzeText()}
                  disabled={isAnalyzing || !inputText.trim()}
                  className={`w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center space-x-2 transition-all ${
                    isAnalyzing || !inputText.trim()
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-rose-500 via-rose-600 to-amber-600 hover:from-rose-600 hover:to-amber-700 text-white shadow-lg shadow-rose-500/20 active:scale-[0.99]'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-white" />
                      <span>Detecting Language & Extracting Flags...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Analyze Communication Flags</span>
                    </>
                  )}
                </button>
              </div>

              {/* Presets List */}
              <div className="glass-panel p-4 rounded-2xl border border-slate-800">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
                  <Globe className="w-3.5 h-3.5 text-rose-400" />
                  <span>Multilingual Benchmarks</span>
                </h3>
                <div className="space-y-1.5">
                  {samples.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => loadPreset(s)}
                      className="w-full text-left p-2.5 rounded-xl hover:bg-slate-900 border border-transparent hover:border-slate-800 transition-all flex items-center justify-between group"
                    >
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-medium text-slate-200 group-hover:text-rose-300 transition-colors">
                            {s.title}
                          </span>
                        </div>
                        <span className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">
                          {s.text}
                        </span>
                      </div>
                      <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-all flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Column: Analysis Results */}
            <div className="lg:col-span-7 space-y-6">
              {result ? (
                <>
                  {/* Hero Verdict Card */}
                  {(() => {
                    const theme = getFlagTheme(result.summary.flag);
                    return (
                      <div className={`p-6 rounded-2xl border ${theme.bg} ${theme.glow} relative overflow-hidden transition-all`}>
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                          <div className="flex items-center space-x-4">
                            <div className="p-3 rounded-2xl bg-slate-950/60 border border-white/10">
                              {theme.icon}
                            </div>
                            <div>
                              <div className="flex items-center space-x-2">
                                <span className={`text-xs px-2.5 py-0.5 rounded-full font-mono uppercase tracking-wider ${theme.badge}`}>
                                  {result.summary.flag}
                                </span>
                                <span className="text-xs text-slate-400 font-mono">
                                  {result.summary.confidence}% Confidence
                                </span>
                                {result.language_detection && (
                                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-900/90 text-slate-300 border border-slate-700 flex items-center space-x-1">
                                    <Globe className="w-3 h-3 text-rose-400" />
                                    <span>{result.language_detection.language}</span>
                                  </span>
                                )}
                              </div>
                              <h2 className="text-2xl font-extrabold text-white mt-1">
                                {result.summary.behavior_category}
                              </h2>
                            </div>
                          </div>

                          <div className="flex items-center space-x-2 self-end sm:self-auto">
                            <button
                              onClick={copyAnalysis}
                              className="px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-xs text-slate-300 flex items-center space-x-1.5 transition-all"
                            >
                              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                              <span>{copied ? 'Copied' : 'Share'}</span>
                            </button>
                          </div>
                        </div>

                        {/* Confidence Progress Bar */}
                        <div className="mt-5 pt-4 border-t border-white/10">
                          <div className="flex justify-between text-xs text-slate-400 mb-1">
                            <span>Behavioral Risk Severity</span>
                            <span className="font-mono">{Math.round(result.summary.severity_score * 100)}%</span>
                          </div>
                          <div className="w-full h-2 bg-slate-950/70 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700"
                              style={{
                                width: `${Math.max(6, result.summary.severity_score * 100)}%`,
                                backgroundColor: theme.color
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })()}

                  {/* Active Self-Training & Adaptation Bar */}
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                      <div className="flex items-center space-x-2">
                        <Brain className="w-4 h-4 text-amber-400" />
                        <span className="text-xs font-semibold text-slate-200">
                          Teach the Model (Continuous Self-Training):
                        </span>
                      </div>
                      <span className="text-[11px] text-slate-400">
                        Is this flag verdict accurate for your dialect?
                      </span>
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <button
                        onClick={() => handleFeedback(result.summary.flag)}
                        disabled={isRetraining}
                        className="px-3 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 text-xs font-medium flex items-center space-x-1 transition-all"
                      >
                        <ThumbsUp className="w-3 h-3" />
                        <span>Confirm Verdict & Retrain</span>
                      </button>

                      <span className="text-xs text-slate-600">or Correct to:</span>

                      <button
                        onClick={() => handleFeedback('Red Flag')}
                        disabled={isRetraining}
                        className="px-2.5 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-300 text-xs font-medium transition-all"
                      >
                        🔴 Red Flag
                      </button>

                      <button
                        onClick={() => handleFeedback('Yellow Flag')}
                        disabled={isRetraining}
                        className="px-2.5 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 text-xs font-medium transition-all"
                      >
                        🟡 Yellow Flag
                      </button>

                      <button
                        onClick={() => handleFeedback('Green Flag')}
                        disabled={isRetraining}
                        className="px-2.5 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 text-xs font-medium transition-all"
                      >
                        🟢 Green Flag
                      </button>
                    </div>

                    {feedbackSuccess && (
                      <div className="mt-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center space-x-2">
                        <Check className="w-3.5 h-3.5 text-amber-400" />
                        <span>{feedbackSuccess}</span>
                      </div>
                    )}
                  </div>

                  {/* Explain WHY: The Star Feature */}
                  <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                    <div className="flex items-center space-x-2">
                      <Sparkles className="w-5 h-5 text-amber-400" />
                      <h3 className="text-base font-bold text-white">
                        Why was this flagged? (Explainability Engine)
                      </h3>
                    </div>

                    <p className="text-sm text-slate-300 leading-relaxed bg-slate-900/60 p-4 rounded-xl border border-slate-800">
                      {result.explain_why.summary_reason}
                    </p>

                    {/* Detected Patterns */}
                    {result.explain_why.detected_patterns?.length > 0 && (
                      <div>
                        <span className="text-xs font-semibold text-slate-400 block mb-2">
                          Identified Communication Dynamics:
                        </span>
                        <div className="flex flex-wrap gap-2">
                          {result.explain_why.detected_patterns.map((pat, idx) => (
                            <span
                              key={idx}
                              className="text-xs px-3 py-1 rounded-full bg-slate-900 border border-slate-700 text-slate-200 font-medium"
                            >
                              • {pat}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Highlighted Triggers */}
                    {result.explain_why.highlighted_triggers?.length > 0 && (
                      <div>
                        <span className="text-xs font-semibold text-slate-400 block mb-2">
                          Linguistic Trigger Phrases ({result.language_detection?.language || 'Multilingual'}):
                        </span>
                        <div className="flex flex-wrap gap-2">
                          {result.explain_why.highlighted_triggers.map((trig, idx) => (
                            <span
                              key={idx}
                              className="text-xs px-2.5 py-1 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 font-mono"
                            >
                              "{trig}"
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Top ML Subword & Word Features */}
                    {result.contributing_features?.length > 0 && (
                      <div className="pt-2">
                        <span className="text-[11px] font-mono text-slate-500 block mb-1.5">
                          Top Statistical Subword & N-Gram Weights:
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {result.contributing_features.map((item, idx) => (
                            <span
                              key={idx}
                              className="text-[11px] px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-400 font-mono"
                            >
                              {item.feature} (+{item.weight})
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Multi-Class Probabilities & Sentiment/Toxicity Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Probabilities */}
                    <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
                        <BarChart3 className="w-3.5 h-3.5 text-rose-400" />
                        <span>Class Probability Distribution</span>
                      </h4>
                      <div className="space-y-2">
                        {Object.entries(result.probabilities).map(([cls, pct]) => {
                          let barColor = 'bg-slate-500';
                          if (cls === 'Red Flag') barColor = 'bg-rose-500';
                          else if (cls === 'Yellow Flag') barColor = 'bg-amber-500';
                          else if (cls === 'Green Flag') barColor = 'bg-emerald-500';
                          return (
                            <div key={cls}>
                              <div className="flex justify-between text-xs text-slate-300 mb-1">
                                <span>{cls}</span>
                                <span className="font-mono text-slate-400">{pct}%</span>
                              </div>
                              <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${barColor}`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Toxicity & Sentiment */}
                    <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
                        <Flame className="w-3.5 h-3.5 text-rose-400" />
                        <span>Emotional Tone & Toxicity</span>
                      </h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-xs text-slate-300 mb-1">
                            <span>Toxicity Risk Index</span>
                            <span className="font-mono font-semibold text-rose-400">
                              {result.sentiment_and_toxicity.toxicity_score}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500 rounded-full"
                              style={{ width: `${result.sentiment_and_toxicity.toxicity_score}%` }}
                            />
                          </div>
                        </div>

                        <div className="flex items-center justify-between pt-1">
                          <span className="text-xs text-slate-400">Emotional Polarity</span>
                          <span className="text-xs font-medium text-slate-200">
                            {result.sentiment_and_toxicity.sentiment_label}
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-xs text-slate-400">Linguistic Intensity</span>
                          <span className="text-xs font-medium text-slate-200">
                            {result.sentiment_and_toxicity.emotional_intensity}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Turn-by-Turn Message Breakdown */}
                  {result.conversation_breakdown?.messages?.length > 0 && (
                    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
                          <MessageSquare className="w-4 h-4 text-rose-400" />
                          <span>Conversation Breakdown ({result.conversation_breakdown.total_turns} turns)</span>
                        </h3>
                        <span className="text-xs text-slate-400 font-mono">
                          {result.language_detection?.language}
                        </span>
                      </div>

                      <div className="space-y-2.5 max-h-80 overflow-y-auto pr-1">
                        {result.conversation_breakdown.messages.map((m) => {
                          const mTheme = getFlagTheme(m.flag);
                          return (
                            <div
                              key={m.id}
                              className={`p-3.5 rounded-xl border transition-all ${
                                m.is_concerning
                                  ? 'bg-rose-500/5 border-rose-500/30'
                                  : 'bg-slate-900/60 border-slate-800'
                              }`}
                            >
                              <div className="flex items-center justify-between mb-1.5">
                                <span className="text-xs font-semibold text-slate-300">
                                  {m.sender}
                                </span>
                                <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${mTheme.badge}`}>
                                  {m.flag} ({m.confidence}%)
                                </span>
                              </div>
                              <p className="text-xs text-slate-200 font-mono leading-relaxed">
                                {m.text}
                              </p>
                              {m.triggers?.length > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {m.triggers.map((t, i) => (
                                    <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300">
                                      Flag trigger: "{t}"
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Responsible AI Disclaimer Banner */}
                  <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-slate-400 text-xs leading-relaxed flex items-start space-x-3">
                    <HelpCircle className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold text-slate-300 block mb-0.5">
                        Responsible AI & Communication Ethics Disclaimer:
                      </span>
                      {result.explain_why.disclaimer}
                    </div>
                  </div>
                </>
              ) : (
                /* Empty state when no analysis yet */
                <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center space-y-4">
                  <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 text-rose-400 flex items-center justify-center mx-auto text-2xl">
                    🚩
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">Ready for Multilingual Analysis</h3>
                    <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
                      Paste a chat thread in Hinglish, Marathi, Hindi, or English, upload a screenshot, or click one of the benchmark scenarios.
                    </p>
                  </div>
                  <button
                    onClick={() => samples[0] && loadPreset(samples[0])}
                    className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-slate-200 transition-all"
                  >
                    <span>Run Hinglish Red Flag Benchmark</span>
                    <ArrowRight className="w-3.5 h-3.5 text-rose-400" />
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-slate-200">REDFLAG AI</span>
            <span>•</span>
            <span>Multilingual NLP + Active Self-Training Architecture</span>
          </div>
          <div>
            Supports English, Hinglish, Hindi, Marathi & Continuous Online Learning
          </div>
        </div>
      </footer>
    </div>
  );
}
