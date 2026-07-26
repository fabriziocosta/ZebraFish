import { useEffect, useMemo, useState } from "react";
import { fetchInvestigation } from "./api";
import type { Candidate, Evidence } from "./model";
import { ActiveHypothesisCard, AlertPanel, BeliefTimeline, CurrentExperimentCard, DecisionQueue, DetailDrawer, EvidencePanel, ExpectedOutcomes, FocusedGraph, InvestigationHeader } from "./components";
import "./styles.css";

const campaignOptions = ["cnn", "transformer"];

export default function App() {
  const [campaign, setCampaign] = useState(new URLSearchParams(window.location.search).get("campaign") || "cnn");
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
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchInvestigation>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set());
  const [detail, setDetail] = useState<{ title: string; value: unknown } | null>(null);

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
  }, [campaign, view, level, relationDepth, entityType, relationType]);

  useEffect(() => {
    window.localStorage.setItem("mission-control-theme", dark ? "dark" : "light");
  }, [dark]);

  const titleDetail = (title: string, value: unknown) => setDetail({ title, value });
  const theme = dark ? "theme-dark" : "theme-light";
  const hasData = Boolean(data);
  const diagnostics = useMemo(() => data?.diagnostics.errors || [], [data]);

  const cardsCommand = (open: boolean) => window.dispatchEvent(new CustomEvent("mission-control:cards", { detail: { open } }));
  return <main className={theme}>
    <nav className="topbar"><div className="brand-mark">ZF</div><div className="nav-controls"><label>campaign <select value={campaign} onChange={(event) => setCampaign(event.target.value)}>{campaignOptions.map((option) => <option key={option}>{option}</option>)}</select></label><div className="view-tabs">{["current", "history", "full"].map((option) => <button className={view === option ? "selected" : ""} key={option} onClick={() => setView(option)}>{option}</button>)}</div><div className="collapse-controls"><button onClick={() => cardsCommand(false)}>collapse all</button><button onClick={() => cardsCommand(true)}>expand all</button></div><button className="theme-toggle" onClick={() => setDark((value) => !value)} aria-label="Toggle theme">{dark ? "☼" : "☾"}</button></div></nav>
    {loading && !hasData && <div className="page-state"><div className="loader" /><h2>Reading scientific state…</h2><p>Connecting to the local campaign adapter.</p></div>}
    {error && !hasData && <div className="page-state error-state"><h2>Dashboard unavailable</h2><p>{error}</p><p>Start the API with <code>./run_campaign dashboard {campaign}</code>.</p></div>}
    {data && <div className="page-shell">
      <InvestigationHeader data={data} />
      {diagnostics.length > 0 && <div className="schema-strip"><strong>Degraded data:</strong> {diagnostics.join(" · ")}</div>}
      <div className="story-grid"><ActiveHypothesisCard data={data} onSelect={(id) => titleDetail("Hypothesis", data.active_hypothesis)} /><CurrentExperimentCard data={data} /></div>
      <div className="story-grid"><ExpectedOutcomes data={data} /><DecisionQueue data={data} onSelect={(candidate: Candidate) => titleDetail(candidate.title, candidate)} /></div>
      <EvidencePanel data={data} onSelect={(item: Evidence) => titleDetail(item.type.replaceAll("_", " "), item)} />
      <div className="two-column"><BeliefTimeline data={data} onSelect={(event) => titleDetail("Belief update", event)} /><AlertPanel data={data} acknowledged={acknowledged} onAcknowledge={(id) => setAcknowledged((current) => new Set(current).add(id))} /></div>
      <FocusedGraph data={data} level={level} relationDepth={relationDepth} scale={graphScale} entityType={entityType} relationType={relationType} onLevel={setLevel} onDepth={setRelationDepth} onScale={setGraphScale} onEntityType={setEntityType} onRelationType={setRelationType} onNodeSelect={(node) => titleDetail(String(node.tooltip || node.label || node.id), node)} />
    </div>}
    <DetailDrawer detail={detail} onClose={() => setDetail(null)} />
  </main>;
}
