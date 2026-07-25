import type { ReactNode } from "react";
import type { Alert, Candidate, Evidence, Investigation, MetricPoint } from "./model";

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Number(value).toPrecision(3).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Card({ title, eyebrow, children, className = "" }: { title: string; eyebrow?: string; children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}><div className="card-heading">{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{children}</section>;
}

export function InvestigationHeader({ data }: { data: Investigation }) {
  const statusTone = data.investigation.health === "critical" ? "critical" : data.investigation.health === "warning" ? "warning" : "healthy";
  return <header className="mission-header">
    <div><div className="eyebrow">ZEBRAFISH / MISSION CONTROL</div><h1>{data.project.name}</h1><p className="question-line">{String(data.active_question?.text || data.project.objective || "No active research question")}</p></div>
    <div className="header-stats">
      <div><span className="eyebrow">INVESTIGATION</span><Badge tone={statusTone}>{data.investigation.status}</Badge></div>
      <div><span className="eyebrow">HEALTH</span><Badge tone={statusTone}>{data.investigation.health}</Badge></div>
      <div><span className="eyebrow">BUDGET</span><strong>{formatNumber(data.project.remaining_gpu_hours)} GPU h</strong></div>
      <div><span className="eyebrow">UPDATED</span><strong>{data.investigation.last_updated ? new Date(data.investigation.last_updated).toLocaleTimeString() : "—"}</strong></div>
    </div>
  </header>;
}

export function BeliefScore({ belief }: { belief: Record<string, unknown> | undefined }) {
  const probability = typeof belief?.probability === "number" ? belief.probability : null;
  const previous = typeof belief?.previous_probability === "number" ? belief.previous_probability : null;
  const delta = typeof belief?.delta === "number" ? belief.delta : null;
  return <div className="belief-score">
    <div className="belief-number">{probability === null ? "Not available" : formatNumber(probability * 100)}<small>{probability === null ? "" : "%"}</small></div>
    {probability !== null && <div className="belief-bar"><span style={{ width: `${Math.max(0, Math.min(1, probability)) * 100}%` }} /></div>}
    <div className="belief-meta">Previous: {previous === null ? "—" : `${formatNumber(previous * 100)}%`} · Change: {delta === null ? "—" : `${delta >= 0 ? "+" : ""}${formatNumber(delta * 100)} pp`}</div>
    <div className="muted">{String(belief?.interpretation || "Belief calibration unavailable")}{belief?.confidence ? ` · confidence: ${String(belief.confidence)}` : ""}</div>
  </div>;
}

export function ActiveHypothesisCard({ data, onSelect }: { data: Investigation; onSelect: (id: string) => void }) {
  const hypothesis = data.active_hypothesis;
  if (!hypothesis) return <Card title="Active hypothesis"><div className="empty">No active hypothesis is recorded.</div></Card>;
  const id = String(hypothesis.id || "");
  return <Card title="Active hypothesis" eyebrow="THE SCIENTIFIC CLAIM" className="hero-card">
    <button className="entity-button" onClick={() => onSelect(id)}><h3>{String(hypothesis.title || "Untitled hypothesis")}</h3><p className="statement">{String(hypothesis.statement || hypothesis.text || "No statement recorded")}</p></button>
    <div className="hypothesis-grid"><div><span className="eyebrow">CURRENT BELIEF</span><BeliefScore belief={hypothesis.belief_score as Record<string, unknown>} /></div><div><span className="eyebrow">MECHANISM</span><p>{String(hypothesis.mechanism || "Mechanism not recorded.")}</p><span className="eyebrow">FALSIFICATION</span><p>{Array.isArray(hypothesis.falsification_criteria) ? hypothesis.falsification_criteria.join(" ") : "Criteria not recorded."}</p></div></div>
  </Card>;
}

function Sparkline({ points }: { points: MetricPoint[] }) {
  const values = points.flatMap((point) => [point.train, point.validation, point.primary].filter((value): value is number => typeof value === "number"));
  if (!values.length) return <div className="spark-empty">No time-series metric available</div>;
  const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
  const line = (key: "train" | "validation" | "primary", color: string) => {
    const selected = points.map((point, index) => ({ point: point[key], index })).filter((item): item is { point: number; index: number } => typeof item.point === "number");
    if (!selected.length) return null;
    return <polyline fill="none" stroke={color} strokeWidth="2" points={selected.map(({ point, index }) => `${(index / Math.max(1, points.length - 1)) * 100},${38 - ((point - min) / range) * 32}`).join(" ")} />;
  };
  return <svg className="sparkline" viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label="Metric history">{line("train", "#6aa6ff")}{line("validation", "#ef9b55")}{line("primary", "#67d4a4")}</svg>;
}

export function CurrentExperimentCard({ data }: { data: Investigation }) {
  const experiment = data.current_experiment;
  return <Card title="Current experiment" eyebrow="WHAT IS RUNNING">
    <div className="experiment-title"><div><h3>{experiment.title}</h3><p className="muted">{experiment.id || "No active experiment ID"} · trial {experiment.trial_id || "—"}</p></div><Badge tone={experiment.process_running ? "healthy" : "warning"}>{experiment.status}</Badge></div>
    <p>{experiment.purpose || "Purpose not recorded."}</p>
    <div className="metric-grid"><div><span className="eyebrow">PHASE</span><strong>{experiment.stage || "—"}</strong></div><div><span className="eyebrow">PROGRESS</span><strong>{experiment.current_epoch && experiment.total_epochs ? `${experiment.current_epoch}/${experiment.total_epochs}` : "—"}</strong></div><div><span className="eyebrow">ELAPSED</span><strong>{formatDuration(experiment.elapsed_seconds)}</strong></div><div><span className="eyebrow">ETA</span><strong>{formatDuration(experiment.estimated_remaining_seconds)}</strong></div><div><span className="eyebrow">CURRENT</span><strong>{formatNumber(experiment.current_metric)}</strong></div><div><span className="eyebrow">BEST</span><strong>{formatNumber(experiment.best_metric)}</strong></div></div>
    <div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(1, experiment.progress_fraction || 0)) * 100}%` }} /></div><div className="muted metric-caption">{experiment.primary_metric} · checkpoint {experiment.checkpoint ? "available" : "not recorded"}</div><Sparkline points={experiment.metric_series} />
  </Card>;
}

export function ExpectedOutcomes({ data }: { data: Investigation }) {
  return <Card title="Expected outcomes" eyebrow="WHAT WOULD CHANGE OUR MIND"><div className="outcomes-list">{data.expected_outcomes.length ? data.expected_outcomes.map((outcome) => <div className="outcome-row" key={outcome.id}><div><strong>{outcome.statement}</strong><span className="muted">Active hypothesis prediction</span></div><Badge tone="neutral">{outcome.status}</Badge></div>) : <div className="empty">No pre-registered expected outcomes are available for this experiment.</div>}</div></Card>;
}

function EvidenceColumn({ title, items, tone, onSelect }: { title: string; items: Evidence[]; tone: string; onSelect: (item: Evidence) => void }) {
  return <div className="evidence-column"><div className="column-title"><h3>{title}</h3><Badge tone={tone}>{items.length}</Badge></div>{items.length ? items.slice(0, 8).map((item) => <button className="evidence-item" key={item.id} onClick={() => onSelect(item)}><strong>{item.type.replaceAll("_", " ")}</strong><span>{item.summary || "No summary"}</span><small>{item.source_experiments.join(", ") || "source unavailable"} · reliability {formatNumber(item.reliability)}</small></button>) : <div className="empty">No classified evidence.</div>}</div>;
}

export function EvidencePanel({ data, onSelect }: { data: Investigation; onSelect: (item: Evidence) => void }) {
  return <Card title="Evidence" eyebrow="WHY WE BELIEVE OR DOUBT IT" className="evidence-card"><div className="evidence-grid"><EvidenceColumn title="Supporting" items={data.evidence.supporting} tone="healthy" onSelect={onSelect} /><EvidenceColumn title="Contradicting" items={data.evidence.contradicting} tone="critical" onSelect={onSelect} /></div>{data.evidence.inconclusive.length > 0 && <details className="inconclusive"><summary>{data.evidence.inconclusive.length} inconclusive observations</summary>{data.evidence.inconclusive.slice(0, 8).map((item) => <button className="evidence-item" key={item.id} onClick={() => onSelect(item)}><strong>{item.type.replaceAll("_", " ")}</strong><span>{item.summary}</span></button>)}</details>}</Card>;
}

export function DecisionQueue({ data, onSelect }: { data: Investigation; onSelect: (candidate: Candidate) => void }) {
  return <Card title="Next decisions" eyebrow="AUTONOMOUS POLICY QUEUE"><div className="queue-list">{data.candidates.length ? data.candidates.slice(0, 6).map((candidate) => <button className="queue-item" key={candidate.id} onClick={() => onSelect(candidate)}><div><strong>{candidate.title}</strong><span>{candidate.rationale || candidate.purpose || "No rationale recorded."}</span><small>{candidate.estimated_gpu_hours === null || candidate.estimated_gpu_hours === undefined ? "cost unavailable" : `${formatNumber(candidate.estimated_gpu_hours)} GPU h`} · {candidate.status}</small></div><Badge tone={candidate.status === "rejected" ? "critical" : candidate.status === "running" ? "healthy" : "neutral"}>{candidate.status}</Badge></button>) : <div className="empty">No candidate experiment is currently recorded.</div>}</div><p className="muted queue-note">Candidates are displayed read-only. Valid bounded candidates launch automatically; unsafe candidates are rejected and recorded.</p></Card>;
}

export function BeliefTimeline({ data, onSelect }: { data: Investigation; onSelect: (event: Record<string, unknown>) => void }) {
  return <Card title="Belief history" eyebrow="AUDITABLE UPDATES"><div className="timeline">{data.belief_history.length ? data.belief_history.map((event, index) => <button className="timeline-event" key={String(event.id || index)} onClick={() => onSelect(event)}><span className="timeline-dot" /><div><strong>{String(event.timestamp || "Unknown time")}</strong><span>{String(event.actor || "controller")} · {String(event.rationale || "belief update")}</span></div></button>) : <div className="empty">No explicit belief updates have been recorded.</div>}</div></Card>;
}

export function AlertPanel({ data, acknowledged, onAcknowledge }: { data: Investigation; acknowledged: Set<string>; onAcknowledge: (id: string) => void }) {
  const visible = data.alerts.filter((alert) => !acknowledged.has(alert.id));
  return <Card title="Alerts" eyebrow="ONLY WHEN ACTION IS REQUIRED" className={visible.length ? "alert-card" : ""}>{visible.length ? visible.map((alert: Alert) => <div className={`alert alert-${alert.severity}`} key={alert.id}><div><Badge tone={alert.severity}>{alert.severity}</Badge><strong>{alert.type.replaceAll("_", " ")}</strong><p>{alert.condition}</p><small>Recommended: {alert.recommended_action} · {alert.automatic ? "automatic policy" : "inspection required"}</small></div><button className="text-button" onClick={() => onAcknowledge(alert.id)}>Acknowledge</button></div>) : <div className="empty">No active alerts.</div>}</Card>;
}

export function FocusedGraph({ data, level, relationDepth, onLevel, onDepth }: { data: Investigation; level: number; relationDepth: number; onLevel: (value: number) => void; onDepth: (value: number) => void }) {
  return <Card title="Focused reasoning graph" eyebrow="SECONDARY DRILL-DOWN"><div className="graph-controls"><label>detail <input type="range" min="0" max="5" value={level} onChange={(event) => onLevel(Number(event.target.value))} /></label><label>relations <input type="range" min="0" max="5" value={relationDepth} onChange={(event) => onDepth(Number(event.target.value))} /></label><span className="muted">{data.graph.nodes.length} nodes · {data.graph.edges.length} edges</span></div>{data.graph.svg ? <div className="graph-svg" dangerouslySetInnerHTML={{ __html: data.graph.svg }} /> : <div className="empty">No focused graph evidence is available.</div>}</Card>;
}

export function DetailDrawer({ detail, onClose }: { detail: { title: string; body: string } | null; onClose: () => void }) {
  if (!detail) return null;
  return <div className="drawer-backdrop" onClick={onClose}><aside className="drawer" onClick={(event) => event.stopPropagation()}><button className="close-button" onClick={onClose}>×</button><div className="eyebrow">DETAIL</div><h2>{detail.title}</h2><pre>{detail.body}</pre></aside></div>;
}
