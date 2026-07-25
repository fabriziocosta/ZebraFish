import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Alert, Candidate, Evidence, Investigation, MetricPoint } from "./model";

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Number(value).toPrecision(3).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export function formatText(value: unknown): string {
  return String(value ?? "").replace(/[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?/g, (token) => {
    const number = Number(token);
    return Number.isFinite(number) ? formatNumber(number) : token;
  });
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
  const ref = useRef<HTMLDetailsElement>(null);
  const storageKey = `mission-control-card:${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(storageKey);
      if (ref.current && saved !== null) ref.current.open = saved === "open";
    } catch { /* local storage is an optional preference */ }
    const onCardsCommand = (event: Event) => {
      const command = (event as CustomEvent<{ open: boolean }>).detail;
      if (ref.current && typeof command?.open === "boolean") {
        ref.current.open = command.open;
        try { window.localStorage.setItem(storageKey, command.open ? "open" : "closed"); } catch { /* optional */ }
      }
    };
    window.addEventListener("mission-control:cards", onCardsCommand);
    return () => window.removeEventListener("mission-control:cards", onCardsCommand);
  }, [storageKey]);
  return <details ref={ref} className={`card ${className}`} open onToggle={(event) => { try { window.localStorage.setItem(storageKey, event.currentTarget.open ? "open" : "closed"); } catch { /* optional */ } }}><summary className="card-heading">{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2><span className="collapse-icon" aria-hidden="true">⌃</span></summary><div className="card-body">{children}</div></details>;
}

export function InvestigationHeader({ data }: { data: Investigation }) {
  const statusTone = data.investigation.health === "critical" ? "critical" : data.investigation.health === "warning" ? "warning" : "healthy";
  return <header className="mission-header">
    <div><div className="eyebrow">ZEBRAFISH / MISSION CONTROL</div><h1>{data.project.name}</h1><p className="question-line">{String(data.active_question?.text || data.project.objective || "No active research question")}</p></div>
    <div className="header-stats">
      <div><span className="eyebrow">RUN STATUS</span><Badge tone={data.current_experiment.process_running ? "healthy" : "warning"}>{data.current_experiment.status}</Badge></div>
      <div><span className="eyebrow">SCIENTIFIC HEALTH</span><Badge tone={statusTone}>{data.investigation.health}</Badge></div>
      <div><span className="eyebrow">CONTROLLER</span><Badge tone={data.health.controller_metadata === "stale" ? "warning" : "healthy"}>{data.health.controller_metadata === "stale" ? "metadata stale" : "consistent"}</Badge></div>
      <div><span className="eyebrow">INTERVENTION</span><Badge tone={data.health.intervention_required ? "critical" : "healthy"}>{data.health.intervention_required ? "required" : "not required"}</Badge></div>
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
  const quality = (hypothesis.hypothesis_quality || {}) as Record<string, unknown>;
  const counts = data.evidence.counts;
  const belief = (hypothesis.belief_score || {}) as Record<string, unknown>;
  return <Card title="Active hypothesis" eyebrow="THE SCIENTIFIC CLAIM" className="hero-card">
    <button className="entity-button" onClick={() => onSelect(id)}><div className="inline-badges"><Badge tone={quality.quality === "generic_seed" ? "warning" : quality.quality === "missing" ? "critical" : "healthy"}>{String(quality.quality || "missing").replaceAll("_", " ")}</Badge></div><h3>{formatText(hypothesis.title || "Untitled hypothesis")}</h3><p className="statement">{formatText(hypothesis.statement || hypothesis.text || "No statement recorded")}</p></button>
    {quality.quality === "generic_seed" && <div className="coverage-warning">This is a generic seed hypothesis. Mechanism, scope, or assumptions are not sufficiently specified for a strong scientific interpretation.</div>}
    <div className="hypothesis-grid"><div><span className="eyebrow">CURRENT BELIEF</span><BeliefScore belief={belief} /><div className="coverage-grid"><span>supporting <b>{counts.supporting ?? 0}</b></span><span>contradicting <b>{counts.contradicting ?? 0}</b></span><span>unclassified <b>{counts.unclassified ?? 0}</b></span><span>updates <b>{belief.history_available ? data.belief_history.length : 0}</b></span></div></div><div><span className="eyebrow">MECHANISM</span><p>{formatText(quality.mechanism || "Not recorded.")}</p><span className="eyebrow">SCOPE & ASSUMPTIONS</span><p>{formatText(quality.scope || quality.assumptions || "Not recorded.")}</p><span className="eyebrow">WHAT WOULD CHANGE OUR MIND</span><p>{Array.isArray(quality.falsification_criteria) && quality.falsification_criteria.length ? formatText(quality.falsification_criteria.join(" ")) : "Falsification criteria not recorded."}</p></div></div>
  </Card>;
}

function Sparkline({ points, metric }: { points: MetricPoint[]; metric: string }) {
  const values = points.flatMap((point) => [point.train, point.validation, point.primary].filter((value): value is number => typeof value === "number"));
  if (!values.length) return <div className="spark-empty">No time-series metric available</div>;
  const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
  const line = (key: "train" | "validation" | "primary", color: string) => {
    const selected = points.map((point, index) => ({ point: point[key], index })).filter((item): item is { point: number; index: number } => typeof item.point === "number");
    if (!selected.length) return null;
    return <polyline fill="none" stroke={color} strokeWidth="2" points={selected.map(({ point, index }) => `${(index / Math.max(1, points.length - 1)) * 100},${38 - ((point - min) / range) * 32}`).join(" ")} />;
  };
  return <div><div className="spark-caption"><span>{metric}</span><span className="spark-legend"><i className="legend-primary" /> displayed <i className="legend-train" /> train <i className="legend-validation" /> validation</span></div><svg className="sparkline" viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label={`${metric} history`}>{line("train", "#6aa6ff")}{line("validation", "#ef9b55")}{line("primary", "#67d4a4")}</svg></div>;
}

export function CurrentExperimentCard({ data }: { data: Investigation }) {
  const experiment = data.current_experiment;
  const metric = experiment.metric_display;
  const compute = experiment.compute;
  return <Card title="Current experiment" eyebrow="WHAT IS RUNNING">
    <div className="experiment-title"><div><h3>{experiment.title}</h3><p className="muted">{experiment.id || "No active experiment ID"} · trial {experiment.trial_id || "—"}</p></div><Badge tone={experiment.process_running ? "healthy" : "warning"}>{experiment.status}</Badge></div>
    <p>{formatText(experiment.purpose || "Purpose not recorded.")}</p>
    <div className="metric-grid"><div><span className="eyebrow">PHASE</span><strong>{experiment.stage || "—"}</strong></div><div><span className="eyebrow">PROGRESS</span><strong>{experiment.current_epoch && experiment.total_epochs ? `${experiment.current_epoch}/${experiment.total_epochs}` : "—"}</strong></div><div><span className="eyebrow">ELAPSED</span><strong>{formatDuration(experiment.elapsed_seconds)}</strong></div><div><span className="eyebrow">ETA</span><strong>{formatDuration(experiment.estimated_remaining_seconds)}</strong></div><div><span className="eyebrow">DISPLAYED VALUE</span><strong>{formatNumber(experiment.current_metric)}</strong></div><div><span className="eyebrow">BEST</span><strong>{formatNumber(experiment.best_metric)}</strong></div></div>
    <div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(1, experiment.progress_fraction || 0)) * 100}%` }} /></div><div className="metric-caption"><strong>{metric.role === "diagnostic" ? `Primary metric: ${metric.requested_metric} — unavailable during pretraining` : `Primary metric: ${metric.requested_metric}`}</strong><span className="muted">{metric.role === "diagnostic" ? `Diagnostic shown: ${metric.display_metric === "val_loss" ? "validation loss" : metric.display_metric || "diagnostic series"} — ${metric.direction.replaceAll("_", " ")}` : metric.direction.replaceAll("_", " ")} · checkpoint {experiment.checkpoint ? "available" : "not recorded"}</span></div>{metric.fallback_reason && <div className="metric-warning">{formatText(metric.fallback_reason)}</div>}<div className="compute-row"><span>compute consumed <b>{formatNumber(compute.consumed_gpu_hours)} GPU h</b></span><span>expected <b>{formatNumber(compute.expected_gpu_hours)} GPU h</b></span><span>remaining <b>{formatNumber(compute.remaining_gpu_hours)} GPU h</b></span></div><div className="baseline-note">Baseline comparison: {formatText(experiment.baseline_comparison.reason || "not available")}</div><Sparkline points={experiment.metric_series} metric={metric.display_metric === "val_loss" ? "validation loss" : metric.display_metric || metric.requested_metric} />
  </Card>;
}

export function ExpectedOutcomes({ data }: { data: Investigation }) {
  const outcomes = data.expected_outcomes;
  return <Card title="Expected outcomes" eyebrow="WHAT WOULD CHANGE OUR MIND"><div className={`registration-banner registration-${outcomes.registration_status}`}><strong>{outcomes.registration_status === "missing" ? "This experiment has no registered predictions and cannot currently distinguish competing hypotheses." : outcomes.registration_status === "partial" ? "Predictions or falsification criteria are incomplete." : "Predictions and falsification criteria were registered before the experiment."}</strong></div><div className="outcomes-list">{outcomes.predictions.length ? outcomes.predictions.map((prediction) => <div className="outcome-row" key={prediction.id}><div><strong>{formatText(prediction.statement)}</strong><span className="muted">Hypotheses: {prediction.hypothesis_ids.join(", ") || "not linked"} · {prediction.source.replaceAll("_", " ")}</span></div><Badge tone={prediction.observed_status === "contradicts_prediction" ? "critical" : prediction.observed_status === "matches_prediction" ? "healthy" : "neutral"}>{prediction.observed_status.replaceAll("_", " ")}</Badge></div>) : <div className="empty">No predictions are registered for this experiment.</div>}</div>{outcomes.falsification_criteria.length > 0 && <div className="falsification-list"><span className="eyebrow">FALSIFICATION CRITERIA</span>{outcomes.falsification_criteria.map((criterion) => <p key={criterion}>• {formatText(criterion)}</p>)}</div>}</Card>;
}

function EvidenceColumn({ title, items, tone, onSelect }: { title: string; items: Evidence[]; tone: string; onSelect: (item: Evidence) => void }) {
  return <div className="evidence-column"><div className="column-title"><h3>{title}</h3><Badge tone={tone}>{items.length}</Badge></div>{items.length ? items.slice(0, 8).map((item) => <button className="evidence-item" key={item.id} onClick={() => onSelect(item)}><strong>{item.type.replaceAll("_", " ")}</strong><span>{formatText(item.summary || "No summary")}</span><small>{item.source_experiments.join(", ") || "source unavailable"} · reliability {formatNumber(item.reliability)} · strength {formatNumber(item.evidence_strength)}</small><em>{formatText(item.explanation)}</em></button>) : <div className="empty">No {title.toLowerCase()} evidence is explicitly classified.</div>}</div>;
}

export function EvidencePanel({ data, onSelect }: { data: Investigation; onSelect: (item: Evidence) => void }) {
  const [mobileCategory, setMobileCategory] = useState("supporting");
  const category = (key: string) => <button className={`evidence-tab ${mobileCategory === key ? "selected" : ""}`} onClick={() => setMobileCategory(key)}>{key} <b>{data.evidence.counts[key] ?? 0}</b></button>;
  return <Card title="Evidence" eyebrow="WHY WE BELIEVE OR DOUBT IT" className="evidence-card"><div className="evidence-tabs" role="tablist">{category("supporting")}{category("contradicting")}{category("inconclusive")}{category("unclassified")}</div><div className="evidence-grid"><div className={`evidence-category category-supporting ${mobileCategory === "supporting" ? "mobile-selected" : ""}`}><EvidenceColumn title="Supporting" items={data.evidence.supporting} tone="healthy" onSelect={onSelect} /></div><div className={`evidence-category category-contradicting ${mobileCategory === "contradicting" ? "mobile-selected" : ""}`}><EvidenceColumn title="Contradicting" items={data.evidence.contradicting} tone="critical" onSelect={onSelect} /></div></div><details className={`inconclusive evidence-category category-inconclusive ${mobileCategory === "inconclusive" ? "mobile-selected" : ""}`}><summary>{data.evidence.inconclusive.length} explicitly inconclusive observations</summary>{data.evidence.inconclusive.slice(0, 8).map((item) => <button className="evidence-item" key={item.id} onClick={() => onSelect(item)}><strong>{item.type.replaceAll("_", " ")}</strong><span>{formatText(item.summary)}</span><small>classified via {item.classification_source.replaceAll("_", " ")}</small><em>{formatText(item.explanation)}</em></button>)}</details><details className={`inconclusive unclassified-evidence evidence-category category-unclassified ${mobileCategory === "unclassified" ? "mobile-selected" : ""}`}><summary>{data.evidence.counts.unclassified || data.evidence.unclassified.length} unclassified observations</summary><p className="muted">Unclassified is different from inconclusive: the observation has not been assigned a direction at all. It must not change belief until classified.</p>{data.evidence.unclassified.slice(0, 8).map((item) => <button className="evidence-item" key={item.id} onClick={() => onSelect(item)}><strong>{item.type.replaceAll("_", " ")}</strong><span>{formatText(item.summary)}</span><small>classification unavailable · source {item.source_experiments.join(", ") || "unavailable"}</small><em>{formatText(item.explanation)}</em></button>)}</details></Card>;
}

export function DecisionQueue({ data, onSelect }: { data: Investigation; onSelect: (candidate: Candidate) => void }) {
  const preferred = data.candidates.find((candidate) => candidate.status !== "rejected") || data.candidates[0];
  return <Card title="Next decisions" eyebrow="AUTONOMOUS POLICY QUEUE">{preferred && <div className="preferred-action"><span className="eyebrow">PREFERRED NEXT ACTION</span><strong>{formatText(preferred.title)}</strong><p>{formatText(preferred.rationale || preferred.purpose || "No rationale recorded.")}</p><span className="muted">value/GPU hour {formatNumber(preferred.value_per_gpu_hour)} · information gain {formatNumber(preferred.expected_information_gain)} · scientific value {formatNumber(preferred.scientific_value)}</span></div>}<div className="queue-list">{data.candidates.length ? data.candidates.slice(0, 6).map((candidate) => <button className="queue-item" key={candidate.id} onClick={() => onSelect(candidate)}><div><strong>{formatText(candidate.title)}</strong><span>{formatText(candidate.rationale || candidate.purpose || "No rationale recorded.")}</span><small>{candidate.estimated_gpu_hours === null || candidate.estimated_gpu_hours === undefined ? "cost unavailable" : `${formatNumber(candidate.estimated_gpu_hours)} GPU h`} · {candidate.status}{candidate.validation_reasons.length ? ` · ${formatText(candidate.validation_reasons.join("; "))}` : ""}</small></div><Badge tone={candidate.status === "rejected" ? "critical" : candidate.status === "running" ? "healthy" : "neutral"}>{candidate.status === "proposed" && candidate.id === preferred?.id ? "automatic next" : candidate.status}</Badge></button>) : <div className="empty">No candidate experiment is currently recorded.</div>}</div><p className="muted queue-note">Candidates are displayed read-only. Valid bounded candidates launch automatically; unsafe candidates are rejected and recorded.</p></Card>;
}

export function BeliefTimeline({ data, onSelect }: { data: Investigation; onSelect: (event: Record<string, unknown>) => void }) {
  const belief = (data.active_hypothesis?.belief_score || {}) as Record<string, unknown>;
  const state = String(belief.history_state || "no_belief_model");
  return <Card title="Belief history" eyebrow="AUDITABLE UPDATES"><div className={`belief-history-note belief-state-${state}`}><strong>{state === "auditable_updates" ? "Auditable updates available" : state === "initial_only" ? "Initial belief only; no subsequent updates recorded" : "No belief model recorded"}</strong><span>{state === "initial_only" ? "The current score is a baseline, not evidence of belief change." : ""}</span></div><div className="timeline">{data.belief_history.length ? data.belief_history.map((event, index) => <button className="timeline-event" key={String(event.id || index)} onClick={() => onSelect(event)}><span className="timeline-dot" /><div><strong>{String(event.timestamp || "Unknown time")}</strong><span>{String(event.actor || "controller")} · {String(event.direction || "belief update")} · {String(event.rationale || "belief update")}</span></div></button>) : <div className="empty">No explicit belief updates have been recorded.</div>}</div></Card>;
}

export function AlertPanel({ data, acknowledged, onAcknowledge }: { data: Investigation; acknowledged: Set<string>; onAcknowledge: (id: string) => void }) {
  const visible = data.alerts.filter((alert) => !acknowledged.has(alert.id));
  const metadata = visible.filter((alert) => alert.type === "stale_controller_metadata");
  const actionAlerts = visible.filter((alert) => alert.type !== "stale_controller_metadata");
  const renderAlert = (alert: Alert) => <div className={`alert alert-${alert.severity}`} key={alert.id}><div><Badge tone={alert.severity}>{alert.severity}</Badge><strong>{alert.type.replaceAll("_", " ")}</strong><p>{formatText(alert.condition)}</p><small>Recommended: {formatText(alert.recommended_action)} · {alert.automatic ? "automatic policy" : "inspection required"}</small></div><button className="text-button" onClick={() => onAcknowledge(alert.id)}>Acknowledge</button></div>;
  return <Card title="Alerts" eyebrow="ACTION AND DATA QUALITY" className={actionAlerts.length ? "alert-card" : ""}>{actionAlerts.length ? actionAlerts.map(renderAlert) : <div className="empty">No process or scientific action alerts.</div>}{metadata.length > 0 && <div className="metadata-warning"><span className="eyebrow">LOW-PRIORITY METADATA WARNING</span>{metadata.map(renderAlert)}<p className="muted">This warning describes controller bookkeeping only. Live process and artifact health determine whether the campaign is running.</p></div>}</Card>;
}

export function FocusedGraph({ data, level, relationDepth, scale, entityType, relationType, onLevel, onDepth, onScale, onEntityType, onRelationType, onNodeSelect }: { data: Investigation; level: number; relationDepth: number; scale: number; entityType: string; relationType: string; onLevel: (value: number) => void; onDepth: (value: number) => void; onScale: (value: number) => void; onEntityType: (value: string) => void; onRelationType: (value: string) => void; onNodeSelect: (node: Record<string, unknown>) => void }) {
  const kinds = [...new Set(data.graph.nodes.map((node) => node.kind))].sort();
  const relations = [...new Set(data.graph.edges.map((edge) => edge.relation).filter(Boolean))].sort();
  return <Card title="Focused reasoning graph" eyebrow="SECONDARY DRILL-DOWN"><div className="graph-controls"><label>detail <input type="range" min="0" max="5" value={level} onChange={(event) => onLevel(Number(event.target.value))} /></label><label>relations <input type="range" min="0" max="5" value={relationDepth} onChange={(event) => onDepth(Number(event.target.value))} /></label><label>scale <input aria-label="Scale graph" type="range" min="40" max="100" step="5" value={scale} onChange={(event) => onScale(Number(event.target.value))} /><span>{scale}%</span></label><label>node type <select value={entityType} onChange={(event) => onEntityType(event.target.value)}><option value="">all</option>{kinds.map((kind) => <option key={kind}>{kind}</option>)}</select></label><label>relation <select value={relationType} onChange={(event) => onRelationType(event.target.value)}><option value="">all</option>{relations.map((relation) => <option key={relation}>{relation}</option>)}</select></label><span className="muted">{data.graph.nodes.length} nodes · {data.graph.edges.length} edges</span></div><div className="graph-legend"><span><i className="node-key node-hypothesis" /> hypothesis</span><span><i className="node-key node-observation" /> observation</span><span><i className="node-key node-experiment" /> experiment/stage</span><span>→ relation direction</span></div>{data.graph.svg ? <div className="graph-svg"><div className="graph-canvas" style={{ transform: `scale(${scale / 100})`, transformOrigin: "top left" }} dangerouslySetInnerHTML={{ __html: data.graph.svg }} /></div> : <div className="empty">No focused graph evidence is available.</div>}<div className="graph-node-list"><span className="eyebrow">SELECT A NODE FOR DETAILS</span>{data.graph.nodes.map((node) => <button key={node.id} onClick={() => onNodeSelect(node)}><span className={`node-key node-${node.kind}`} />{node.label}<small>{node.kind}</small></button>)}</div></Card>;
}

export function DetailDrawer({ detail, onClose }: { detail: { title: string; body: string } | null; onClose: () => void }) {
  if (!detail) return null;
  return <div className="drawer-backdrop" onClick={onClose}><aside className="drawer" onClick={(event) => event.stopPropagation()}><button className="close-button" onClick={onClose}>×</button><div className="eyebrow">DETAIL</div><h2>{detail.title}</h2><pre>{detail.body}</pre></aside></div>;
}
