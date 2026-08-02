import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { controlController, fetchInvestigation, runMetaControllerNow } from "./api";
import type { Candidate, Evidence } from "./model";
import { ActiveHypothesisCard, AlertPanel, BeliefTimeline, CurrentExperimentCard, DecisionQueue, DetailDrawer, DomainGuidanceCard, EvidencePanel, ExpectedOutcomes, FocusedGraph, InvestigationHeader, MetaControllerCard } from "./components";
import "./styles.css";

const campaignOptions = ["cnn-v2"];
const defaultAccents = { accent: "#21815c", success: "#187a5b", running: "#ff8e1c", attention: "#7a315b", warning: "#936000", info: "#3e83bd", critical: "#9c2f62" };
const defaultPageBackground = "#edf2f3";
type AccentColors = typeof defaultAccents;
const accentLabels: Array<[keyof typeof defaultAccents, string]> = [["accent", "Accent"], ["success", "Success"], ["running", "Running"], ["attention", "Attention"], ["warning", "Warning"], ["info", "Info"], ["critical", "Critical"]];

function GearIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19.4 13a7.7 7.7 0 0 0 0-2l2-1.55-2-3.46-2.43.98a7.5 7.5 0 0 0-1.73-1L14.9 3h-4l-.34 2.97a7.5 7.5 0 0 0-1.73 1L6.4 5.99l-2 3.46L6.4 11a7.7 7.7 0 0 0 0 2l-2 1.55 2 3.46 2.43-.98a7.5 7.5 0 0 0 1.73 1L10.9 21h4l.34-2.97a7.5 7.5 0 0 0 1.73-1l2.43.98 2-3.46-2-1.55ZM12.9 15.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7Z" /></svg>;
}

export default function App() {
  const [campaign, setCampaign] = useState(new URLSearchParams(window.location.search).get("campaign") || "cnn-v2");
  const [view, setView] = useState("current");
  const [level, setLevel] = useState(3);
  const [relationDepth, setRelationDepth] = useState(1);
  const [graphScale, setGraphScale] = useState(100);
  const [entityType, setEntityType] = useState("");
  const [relationType, setRelationType] = useState("");
  const [dark, setDark] = useState(() => {
    const saved = window.localStorage.getItem("mission-control-theme");
    return saved ? saved === "dark" : (window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true);
  });
  const [setupOpen, setSetupOpen] = useState(false);
  const [accents, setAccents] = useState<AccentColors>(() => {
    try {
      const saved = window.localStorage.getItem("mission-control-accent-colors");
      const parsed = saved ? JSON.parse(saved) : {};
      return { ...defaultAccents, ...(parsed && typeof parsed === "object" ? parsed : {}) } as AccentColors;
    } catch { return defaultAccents; }
  });
  const [pageBackground, setPageBackground] = useState(() => window.localStorage.getItem("mission-control-page-background") || defaultPageBackground);
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchInvestigation>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set());
  const [detail, setDetail] = useState<{ title: string; value: unknown } | null>(null);
  const [controlMessage, setControlMessage] = useState<string | null>(null);
  const [metaRunPending, setMetaRunPending] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const next = await fetchInvestigation(campaign, { view, level, relationDepth, entityType, relationType });
        if (!cancelled) { setData(next); setError(null); setLoading(false); }
      } catch (err) {
        if (!cancelled) { setError(err instanceof Error ? err.message : String(err)); setLoading(false); }
      }
    };
    setLoading(true); void load();
    const timer = window.setInterval(load, 10000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [campaign, view, level, relationDepth, entityType, relationType, refreshToken]);

  useEffect(() => {
    window.localStorage.setItem("mission-control-theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    window.localStorage.setItem("mission-control-accent-colors", JSON.stringify(accents));
  }, [accents]);
  useEffect(() => {
    window.localStorage.setItem("mission-control-page-background", pageBackground);
  }, [pageBackground]);

  const titleDetail = (title: string, value: unknown) => setDetail({ title, value });
  const handleControl = async (controller: "meta" | "campaign", action: "start" | "stop" | "continue") => {
    if (action === "stop" && !window.confirm(`Stop the ${controller} controller for ${campaign}?`)) return;
    try {
      const result = await controlController(campaign, controller, action);
      setControlMessage(`${controller} ${action}: ${String(result.status || "requested")}`);
      setTimeout(() => setControlMessage(null), 5000);
    } catch (err) {
      setControlMessage(err instanceof Error ? err.message : String(err));
    }
  };
  const handleMetaRunNow = async () => {
    if (metaRunPending) return;
    setMetaRunPending(true);
    setControlMessage("Meta-controller run now: contacting the controller…");
    try {
      const result = await runMetaControllerNow(campaign);
      const status = String(result.status || "requested");
      const detail = status === "requested"
        ? "The controller was woken and is starting a bounded run."
        : status === "started"
          ? "A bounded meta-controller run was started."
          : status === "already_running"
            ? "A meta-controller run is already in progress."
            : `The controller returned ${status}.`;
      setControlMessage(`Meta-controller run now: ${detail} The dashboard will refresh automatically.`);
      setRefreshToken((current) => current + 1);
    } catch (err) {
      setControlMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setMetaRunPending(false);
    }
  };
  const theme = dark ? "theme-dark" : "theme-light";
  const accentStyle = { ...Object.fromEntries(Object.entries(accents).map(([key, value]) => [`--${key === "info" ? "blue" : key}`, value])), "--page-bg": pageBackground } as CSSProperties;
  const hasData = Boolean(data);
  const diagnostics = useMemo(() => data?.diagnostics.errors || [], [data]);

  const cardsCommand = (open: boolean) => window.dispatchEvent(new CustomEvent("mission-control:cards", { detail: { open } }));
  const dashboardControls = <div className="dashboard-toolbar" aria-label="Dashboard controls"><label>campaign <select value={campaign} onChange={(event) => setCampaign(event.target.value)}>{campaignOptions.map((option) => <option key={option}>{option}</option>)}</select></label><div className="view-tabs">{["current", "history", "full"].map((option) => <button className={view === option ? "selected" : ""} key={option} onClick={() => setView(option)}>{option}</button>)}</div><div className="collapse-controls"><button title="Collapse every dashboard card." data-tooltip="Collapse all cards." onClick={() => cardsCommand(false)}>collapse all</button><button title="Expand every dashboard card." data-tooltip="Expand all cards." onClick={() => cardsCommand(true)}>expand all</button></div><button className="setup-button" onClick={() => setSetupOpen(true)} aria-label="Open dashboard setup" title="Open dashboard setup" data-tooltip="Open dashboard setup"><GearIcon /></button></div>;
  return <main className={theme} style={accentStyle}>
    {loading && !hasData && <div className="page-state"><div className="loader" /><h2>Reading scientific state…</h2><p>Connecting to the local campaign adapter.</p></div>}
    {error && !hasData && <div className="page-state error-state"><h2>Dashboard unavailable</h2><p>{error}</p><p>Start the API with <code>./run_campaign dashboard {campaign}</code>.</p></div>}
    {data && <div className="page-shell">
      <InvestigationHeader data={data} controls={dashboardControls} />
      {controlMessage && <div className="control-message" role="status" aria-live="polite">{controlMessage}</div>}
      <MetaControllerCard data={data} onControl={handleControl} onRunNow={handleMetaRunNow} runNowPending={metaRunPending} />
      {diagnostics.length > 0 && <div className="schema-strip"><strong>Degraded data:</strong> {diagnostics.join(" · ")}</div>}
      <div className="story-grid"><ActiveHypothesisCard data={data} onSelect={(id) => titleDetail("Hypothesis", data.active_hypothesis)} /><CurrentExperimentCard data={data} /></div>
      <DomainGuidanceCard data={data} onSelect={titleDetail} />
      <div className="story-grid"><ExpectedOutcomes data={data} /><DecisionQueue data={data} onSelect={(candidate: Candidate) => titleDetail(candidate.title, candidate)} /></div>
      <EvidencePanel data={data} onSelect={(item: Evidence) => titleDetail(item.type.replaceAll("_", " "), item)} />
      <div className="two-column"><BeliefTimeline data={data} onSelect={(event) => titleDetail("Belief update", event)} /><AlertPanel data={data} acknowledged={acknowledged} onAcknowledge={(id) => setAcknowledged((current) => new Set(current).add(id))} /></div>
      <FocusedGraph data={data} level={level} relationDepth={relationDepth} scale={graphScale} entityType={entityType} relationType={relationType} onLevel={setLevel} onDepth={setRelationDepth} onScale={setGraphScale} onEntityType={setEntityType} onRelationType={setRelationType} onNodeSelect={(node) => titleDetail(String(node.label || node.tooltip || node.id).replace(/\s*\n\s*/g, " ").replace(/\s+/g, " ").trim(), node)} />
    </div>}
    <DetailDrawer detail={detail} onClose={() => setDetail(null)} />
    {setupOpen && <div className="setup-backdrop" role="presentation" onClick={() => setSetupOpen(false)}><section className="setup-dialog" role="dialog" aria-modal="true" aria-labelledby="setup-title" onClick={(event) => event.stopPropagation()}><div className="setup-dialog-heading"><div><span className="eyebrow">DASHBOARD PREFERENCES</span><h2 id="setup-title">Setup</h2></div><button className="close-button" onClick={() => setSetupOpen(false)} aria-label="Close setup">×</button></div><div className="setup-section"><span className="eyebrow">APPEARANCE</span><div className="theme-choice"><button className={!dark ? "selected" : ""} onClick={() => setDark(false)}>Light</button><button className={dark ? "selected" : ""} onClick={() => setDark(true)}>Dark</button></div></div><div className="setup-section background-section"><div><span className="eyebrow">PAGE BACKGROUND</span><p className="muted">Choose the gray tone used behind the dashboard in light mode.</p></div><label className="background-choice"><input type="color" value={pageBackground} aria-label="Page background color" onChange={(event) => setPageBackground(event.target.value)} /><code>{pageBackground}</code></label></div><div className="setup-section"><span className="eyebrow">SEMANTIC ACCENTS</span><p className="muted">Running is the color used by Running, Calibrating, Pending, and Awaiting status pills.</p><div className="accent-grid">{accentLabels.map(([key, label]) => <label key={key}><span>{label}</span><input type="color" value={accents[key]} aria-label={`${label} color`} onChange={(event) => setAccents((current) => ({ ...current, [key]: event.target.value }))} /><code>{accents[key]}</code></label>)}</div></div><div className="setup-actions"><button className="secondary-button" onClick={() => { setAccents(defaultAccents); setPageBackground(defaultPageBackground); }}>Reset colors</button><button className="primary-button" onClick={() => setSetupOpen(false)}>Done</button></div></section></div>}
  </main>;
}
