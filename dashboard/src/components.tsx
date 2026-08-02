import { useEffect, useRef, useState, type MouseEvent, type ReactNode } from "react";
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
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
}

type ExperimentAttention = { tone: "error" | "warning"; title: string; detail: string };

function currentExperimentAttention(data: Investigation): ExperimentAttention[] {
  const experiment = data.current_experiment;
  const attention: ExperimentAttention[] = [];
  for (const error of data.diagnostics.errors.slice(0, 3)) {
    attention.push({ tone: "error", title: "Execution error", detail: formatText(error) });
  }
  const watchdog = data.health.watchdog || {};
  if (watchdog.process_identity_mismatch === true) {
    attention.push({ tone: "warning", title: "Process check needs review", detail: "The live process does not exactly match the process recorded at launch." });
  }
  if (experiment.metric_display.role === "diagnostic" && experiment.metric_display.fallback_reason) {
    attention.push({ tone: "warning", title: "Primary metric is not available", detail: `Showing ${experiment.metric_display.display_metric || "a diagnostic metric"} instead of ${experiment.metric_display.requested_metric}.` });
  }
  if (experiment.replicates.status === "legacy_single_seed") {
    attention.push({ tone: "warning", title: "Single-seed run", detail: "Replicate and lockbox evidence are not available for this run." });
  }
  if (experiment.artifact_freshness.status === "stale") {
    attention.push({ tone: "warning", title: "Results may be stale", detail: "The latest metric artifact has not been updated recently." });
  }
  return attention;
}

function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  const normalizedTone = tone === "healthy" || tone === "success" ? "healthy"
    : ["warning", "critical", "attention", "failed", "error", "blocked", "rejected"].includes(tone) ? "attention"
      : ["running", "calibrating", "pending", "awaiting", "warming_up"].includes(tone) ? "running"
        : tone;
  return <span className={`badge badge-${normalizedTone}`}>{children}</span>;
}

function lifecycleTone(status: string, active = false): string {
  const normalized = status.toLowerCase().replaceAll("-", "_");
  if (active || ["running", "starting", "in_progress", "calibrating", "pending", "awaiting", "warming_up"].includes(normalized)) return "running";
  if (["completed", "complete", "pass", "passed", "eligible", "healthy", "consistent", "confirmed", "applied", "recorded", "available"].includes(normalized)) return "healthy";
  if (["not_started", "not_evaluated", "unknown", "unavailable", "neutral"].includes(normalized)) return "neutral";
  return "attention";
}

function Card({ title, eyebrow, summary, children, className = "" }: { title: string; eyebrow?: string; summary?: ReactNode; children: ReactNode; className?: string }) {
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
  return <details ref={ref} className={`card ${className}`} open onToggle={(event) => { try { window.localStorage.setItem(storageKey, event.currentTarget.open ? "open" : "closed"); } catch { /* optional */ } }}><summary className="card-heading">{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2>{summary && <span className="card-summary">{summary}</span>}<span className="collapse-icon" aria-hidden="true">⌃</span></summary><div className="card-body">{children}</div></details>;
}

export function InvestigationHeader({ data, controls }: { data: Investigation; controls?: ReactNode }) {
  const statusTone = data.investigation.health === "healthy" ? "healthy" : "attention";
  return <header className="mission-header">
    <div className="mission-title"><h1><span>Commutative Representation</span><small>Zebrafish experiment</small></h1></div>
    <div className="mission-header-right">{controls}<div className="header-stats">
      <div><span className="eyebrow">RUN STATUS</span><Badge tone={lifecycleTone(data.current_experiment.status, data.current_experiment.process_running)}>{data.current_experiment.status}</Badge></div>
      <div><span className="eyebrow">SCIENTIFIC HEALTH</span><Badge tone={statusTone}>{data.investigation.health}</Badge></div>
      <div><span className="eyebrow">CONTROLLER</span><Badge tone={data.health.controller_metadata === "stale" ? "warning" : "healthy"}>{data.health.controller_metadata === "stale" ? "metadata stale" : "consistent"}</Badge></div>
      <div><span className="eyebrow">INTERVENTION</span><Badge tone={data.health.intervention_required ? "critical" : "healthy"}>{data.health.intervention_required ? "required" : "not required"}</Badge></div>
      <div><span className="eyebrow">UPDATED</span><strong>{data.investigation.last_updated ? new Date(data.investigation.last_updated).toLocaleTimeString() : "—"}</strong></div>
    </div></div>
  </header>;
}

export function MetaControllerCard({ data, onControl, onRunNow }: { data: Investigation; onControl: (controller: "meta" | "campaign", action: "start" | "stop" | "continue") => void; onRunNow: () => void }) {
  const meta = data.meta_controller;
  const severityTone = meta.status === "running" || meta.status === "starting" ? "running" : meta.severity === "critical" || meta.severity === "warning" || meta.status === "meta_controller_safe_stop" || meta.status === "verification_failed" ? "attention" : "healthy";
  const latestActions = meta.actions || [];
  return <Card title="Meta-controller" eyebrow="SYSTEM SUPERVISION" className="meta-controller-card" summary={<Badge tone={severityTone}>{meta.status.replaceAll("_", " ")}</Badge>}>
    <div className="meta-controller-heading"><div>{meta.historical_failure && <span className="metadata-inline-warning">historical failure</span>}<p className="meta-summary">{formatText(meta.summary)}</p></div><div className="controller-buttons"><span className="muted">Operational controls</span><div><button className="primary-button" title="Run one bounded meta-controller diagnosis now." data-tooltip="Run one bounded diagnosis now." onClick={onRunNow}>run now</button><button title="Start the persistent meta-controller loop." data-tooltip="Start the persistent meta-controller loop." onClick={() => onControl("meta", "start")}>start</button><button title="Continue the meta-controller loop after a stop or pause." data-tooltip="Continue the meta-controller loop." onClick={() => onControl("meta", "continue")}>continue</button><button className="danger-button" title="Stop the persistent meta-controller loop." data-tooltip="Stop the meta-controller loop." onClick={() => onControl("meta", "stop")}>stop</button></div></div></div>
    <div className="meta-controller-grid"><div><span className="eyebrow">LAST RUN</span><strong>{meta.last_run_at ? new Date(meta.last_run_at).toLocaleString() : "—"}</strong><small>{meta.last_invocation_source?.replaceAll("_", " ") || "—"}</small></div><div><span className="eyebrow">NEXT RUN</span><strong>{meta.next_run_at ? new Date(meta.next_run_at).toLocaleString() : meta.running ? "After current cycle" : "—"}</strong>{meta.running && !meta.next_run_at && <small>first scheduled cycle is running</small>}</div><div><span className="eyebrow">MANDATE</span><strong>{meta.mandate_version ? meta.mandate_version.slice(0, 20) : "—"}</strong></div><div><span className="eyebrow">ARCHITECTURE</span><strong>{meta.architecture_version ? meta.architecture_version.slice(0, 20) : "—"}</strong></div><div><span className="eyebrow">ROLLBACK</span><strong>{meta.rollback_available ? "available" : "not available"}</strong></div></div>
    {meta.findings.length > 0 && <section className="meta-section"><span className="eyebrow">WHAT IT FOUND</span>{meta.findings.map((finding, index) => <p key={`${finding}-${index}`}>{formatText(finding)}</p>)}</section>}
    <section className="meta-section"><span className="eyebrow">ACTIONS TAKEN</span>{latestActions.length ? latestActions.map((action, index) => <div className="meta-action" key={`${String(action.summary || action.kind)}-${index}`}><Badge tone={action.status === "applied" || action.status === "recorded" ? "healthy" : "attention"}>{String(action.status || "recorded").replaceAll("_", " ")}</Badge><span>{formatText(action.summary || action.kind || "No action")}</span></div>) : <p className="muted">No remediation action was taken.</p>}</section>
    {meta.proposal_only_changes.length > 0 && <section className="meta-section"><span className="eyebrow">PROPOSAL ONLY</span>{meta.proposal_only_changes.map((item, index) => <p key={`${item}-${index}`}>{formatText(item)}</p>)}</section>}
    {meta.unresolved_risks.length > 0 && <section className="meta-section"><span className="eyebrow">UNRESOLVED RISKS</span>{meta.unresolved_risks.map((item, index) => <p key={`${item}-${index}`} className="risk-line">{formatText(item)}</p>)}</section>}
    <div className="meta-controller-footer"><span>Campaign controls affect execution only; scientific candidates remain autonomous.</span><div><button title="Start the selected campaign and its autonomous execution loop." data-tooltip="Start the selected campaign." onClick={() => onControl("campaign", "start")}>start campaign</button><button title="Continue the selected campaign after a stop or pause." data-tooltip="Continue the selected campaign." onClick={() => onControl("campaign", "continue")}>continue campaign</button><button className="danger-button" title="Stop the selected campaign and its active execution." data-tooltip="Stop the selected campaign." onClick={() => onControl("campaign", "stop")}>stop campaign</button></div></div>
  </Card>;
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

function movingAverage(points: Array<{ epoch: number; value: number }>, window = 5) {
  return points.map((point, index) => {
    const start = Math.max(0, index - window + 1);
    const windowValues = points.slice(start, index + 1).map((item) => item.value);
    return { epoch: point.epoch, value: windowValues.reduce((sum, value) => sum + value, 0) / windowValues.length };
  });
}

function MetricPlot({ experiment }: { experiment: Investigation["current_experiment"] }) {
  const metric = experiment.metric_display;
  const plot = experiment.metric_plot;
  const freshness = String(experiment.artifact_freshness.status || "unknown");
  const [windowMode, setWindowMode] = useState<"all" | "recent">("all");
  const [smoothed, setSmoothed] = useState(true);
  const [showComparisons, setShowComparisons] = useState(false);
  const [focused, setFocused] = useState(false);
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  const sourcePoints = windowMode === "recent" ? experiment.metric_series.slice(-30) : experiment.metric_series;
  const validPoints = sourcePoints.filter((point) => [point.primary, point.displayed, point.train, point.validation].some((value) => typeof value === "number"));
  if (!validPoints.length) return <div className="metric-plot-empty"><strong>No time-series metric available</strong><span>The run has not written a history CSV with a plottable metric yet.</span></div>;

  const width = 720;
  const height = 270;
  const margin = { top: 25, right: 20, bottom: 42, left: 52 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const xMin = validPoints[0].epoch;
  const xMax = Math.max(xMin + 1, validPoints[validPoints.length - 1].epoch);
  const displayedKey = metric.role === "primary" ? "primary" : "displayed";
  const currentValues = validPoints.flatMap((point) => [point[displayedKey], point.train, point.validation].filter((value): value is number => typeof value === "number"));
  const comparisonValues = showComparisons ? plot.comparisons.flatMap((comparison) => comparison.points.flatMap((point) => [point[displayedKey], point.validation].filter((value): value is number => typeof value === "number"))) : [];
  const values = [...currentValues, ...comparisonValues];
  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  const range = dataMax - dataMin || Math.max(Math.abs(dataMax) * 0.08, 0.1);
  const yMin = Math.min(plot.y_min ?? dataMin, dataMin) - range * 0.08;
  const yMax = Math.max(plot.y_max ?? dataMax, dataMax) + range * 0.08;
  const yRange = yMax - yMin || 1;
  const x = (epoch: number) => margin.left + ((epoch - xMin) / Math.max(1, xMax - xMin)) * chartWidth;
  const y = (value: number) => margin.top + chartHeight - ((value - yMin) / yRange) * chartHeight;
  const pointSet = (key: "primary" | "displayed" | "train" | "validation", points = validPoints) => points.map((point) => ({ epoch: point.epoch, value: point[key] })).filter((point): point is { epoch: number; value: number } => typeof point.value === "number");
  const series = [
    ...(metric.role === "primary" ? [{ key: "primary" as const, label: `primary metric · ${metric.requested_metric}`, color: "#2e9b78", dash: "" }] : []),
    ...(pointSet("train").length ? [{ key: "train" as const, label: "train", color: "#4f8fe8", dash: "6 4" }] : []),
    ...(pointSet("validation").length ? [{ key: "validation" as const, label: "validation", color: "#e58b45", dash: "" }] : []),
    ...(metric.role !== "primary" && !pointSet("validation").length && pointSet("displayed").length ? [{ key: "displayed" as const, label: `displayed diagnostic · ${metric.display_metric}`, color: "#2e9b78", dash: "" }] : []),
  ];
  const linePoints = (key: "primary" | "displayed" | "train" | "validation", points = validPoints) => {
    const raw = pointSet(key, points);
    const valuesForLine = smoothed && key !== "primary" ? movingAverage(raw) : raw;
    return valuesForLine.map((point) => `${x(point.epoch)},${y(point.value)}`).join(" ");
  };
  const tickValues = Array.from({ length: 4 }, (_, index) => yMin + ((yMax - yMin) * index) / 3);
  const eventMarkers = plot.events.filter((event) => event.epoch >= xMin && event.epoch <= xMax);
  const latestEpoch = validPoints[validPoints.length - 1].epoch;
  const handleMove = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const relativeX = ((event.clientX - rect.left) / rect.width) * width;
    setHoverEpoch(Math.round(xMin + ((relativeX - margin.left) / chartWidth) * (xMax - xMin)));
  };
  return <div className={`metric-plot ${focused ? "metric-plot-focused" : ""}`}>
    <div className="plot-heading"><div><strong>{metric.role === "primary" ? metric.requested_metric : `${metric.display_metric || "Diagnostic metric"} · diagnostic`}</strong><span className="plot-subtitle">{metric.role === "primary" ? "primary objective" : `not the primary objective · ${metric.direction.replaceAll("_", " ")}`}</span><span className={`plot-freshness freshness-${freshness}`}>data {freshness === "unknown" ? "freshness unknown" : freshness}</span></div><div className="plot-actions"><button className={windowMode === "recent" ? "selected" : ""} onClick={() => setWindowMode(windowMode === "recent" ? "all" : "recent")}>recent 30</button><button className={smoothed ? "selected" : ""} onClick={() => setSmoothed(!smoothed)}>{smoothed ? "smoothed" : "raw"}</button>{plot.comparisons.length > 0 && <button className={showComparisons ? "selected" : ""} onClick={() => setShowComparisons(!showComparisons)}>compare prior</button>}<button onClick={() => setFocused(!focused)}>{focused ? "close focus" : "focus chart"}</button></div></div>
    <div className="plot-stat-grid"><div><span>latest</span><strong>{formatNumber(plot.statistics.latest)}</strong></div><div><span>best</span><strong>{formatNumber(plot.statistics.best)}</strong></div><div><span>slope / epoch</span><strong>{plot.statistics.slope === null || plot.statistics.slope === undefined ? "—" : `${plot.statistics.slope >= 0 ? "+" : ""}${formatNumber(plot.statistics.slope)}`}</strong></div><div><span>train–validation gap</span><strong>{formatNumber(plot.statistics.train_validation_gap)}</strong></div></div>
    <div className="plot-frame"><svg className="metric-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${plot.y_axis_label || metric.requested_metric} trajectory`} onMouseMove={handleMove} onMouseLeave={() => setHoverEpoch(null)}>
      {tickValues.map((tick, index) => <g key={tick}><line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} className="plot-gridline" /><text x={margin.left - 9} y={y(tick) + 4} textAnchor="end" className="plot-axis-text">{formatNumber(tick)}</text></g>)}
      <line x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} className="plot-axis" /><line x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} className="plot-axis" />
      <text x={margin.left} y={15} className="plot-axis-title">{plot.y_axis_label || metric.requested_metric}</text><text x={width - margin.right} y={height - 10} textAnchor="end" className="plot-axis-title">epoch</text>
      <text x={margin.left} y={height - 25} className="plot-axis-text">{xMin}</text><text x={width - margin.right} y={height - 25} textAnchor="end" className="plot-axis-text">{xMax}</text>
      {showComparisons && plot.comparisons.map((comparison) => <polyline key={comparison.id} fill="none" stroke="#9aa8ad" strokeWidth="1.5" strokeDasharray="3 4" opacity=".65" points={linePoints(displayedKey, comparison.points)}><title>{comparison.label}</title></polyline>)}
      {series.map((item) => <g key={item.key}><polyline fill="none" stroke={item.color} strokeWidth={item.key === "validation" || item.key === "primary" ? "3" : "2"} strokeDasharray={item.dash} points={linePoints(item.key)}><title>{item.label}</title></polyline>{pointSet(item.key).filter((_, index) => index === pointSet(item.key).length - 1).map((point) => <circle key={`${item.key}-${point.epoch}`} cx={x(point.epoch)} cy={y(point.value)} r="4" fill={item.color}><title>{`${item.label}: ${formatNumber(point.value)} at epoch ${point.epoch}`}</title></circle>)}</g>)}
      {eventMarkers.map((event) => <g key={event.id}><line x1={x(event.epoch)} x2={x(event.epoch)} y1={margin.top} y2={height - margin.bottom} className={`plot-event plot-event-${event.type === "best" ? "best" : "observation"}`}><title>{`${event.label}: ${formatNumber(event.value)} at epoch ${event.epoch}`}</title></line><circle cx={x(event.epoch)} cy={event.type === "best" ? margin.top + 7 : margin.top + 18} r="3" className="plot-event-dot" /></g>)}
      {hoverEpoch !== null && hoverEpoch >= xMin && hoverEpoch <= xMax && <g><line x1={x(hoverEpoch)} x2={x(hoverEpoch)} y1={margin.top} y2={height - margin.bottom} className="plot-hover-line" /><text x={Math.min(width - margin.right - 6, Math.max(margin.left + 6, x(hoverEpoch)))} y={margin.top - 5} textAnchor="middle" className="plot-hover-label">epoch {hoverEpoch}</text></g>}
    </svg></div>
    <div className="plot-legend">{series.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.label}</span>)}{showComparisons && <span><i className="comparison-swatch" />prior trials</span>}</div>
    <p className="plot-interpretation">{formatText(plot.interpretation)}</p>
  </div>;
}

export function CurrentExperimentCard({ data }: { data: Investigation }) {
  const experiment = data.current_experiment;
  const metric = experiment.metric_display;
  const attention = currentExperimentAttention(data);
  const interpretation = formatText(experiment.metric_plot.interpretation || "The current diagnostic trend is not yet interpretable.");
  return <Card title="Current experiment" eyebrow="WHAT IS RUNNING" summary={<Badge tone={lifecycleTone(experiment.status, experiment.process_running)}>{experiment.status}</Badge>}>
    <div className="experiment-identity"><div><h3>{experiment.title}</h3><p className="muted">{formatText(experiment.purpose || "Purpose not recorded.")}</p></div><span className="experiment-stage">{experiment.stage || "—"}</span></div>
    <div className="experiment-stat-grid"><div><span className="eyebrow">PROGRESS</span><strong>{experiment.current_epoch && experiment.total_epochs ? `${experiment.current_epoch}/${experiment.total_epochs}` : "—"}</strong><small>epochs</small></div><div><span className="eyebrow">LATEST</span><strong>{formatNumber(experiment.current_metric)}</strong><small>{metric.display_metric || metric.requested_metric}</small></div><div><span className="eyebrow">BEST</span><strong>{formatNumber(experiment.best_metric)}</strong><small>{metric.display_metric || metric.requested_metric}</small></div><div><span className="eyebrow">ETA</span><strong>{experiment.eta_status === "available" ? formatDuration(experiment.estimated_remaining_seconds) : experiment.eta_status === "warming_up" ? "estimating" : "—"}</strong><small>{formatDuration(experiment.elapsed_seconds)} elapsed</small></div></div>
    <div className="progress-track" aria-label={`Experiment progress ${Math.round(Math.max(0, Math.min(1, experiment.progress_fraction || 0)) * 100)} percent`}><span style={{ width: `${Math.max(0, Math.min(1, experiment.progress_fraction || 0)) * 100}%` }} /></div>
    <details className={`experiment-attention ${attention.length ? "has-attention" : "is-clear"}`} open={attention.length > 0}><summary className="experiment-section-heading"><span className="eyebrow">{attention.length ? "PAY ATTENTION TO" : "STATUS CHECK"}</span><strong>{attention.length ? `${attention.length} item${attention.length === 1 ? "" : "s"}` : "No execution errors reported"}<span className="attention-toggle" aria-hidden="true">⌄</span></strong></summary>{attention.length > 0 && <div className="attention-list">{attention.map((item, index) => <div className={`attention-item attention-${item.tone}`} key={`${item.title}-${index}`}><span className="attention-mark" aria-hidden="true">{item.tone === "error" ? "!" : "i"}</span><div><strong>{item.title}</strong><span>{item.detail}</span></div></div>)}</div>}</details>
    <section className="experiment-takeaway"><span className="eyebrow">WHAT TO TAKE AWAY</span><strong>{metric.role === "diagnostic" ? `${metric.display_metric || "The diagnostic metric"} is the only metric available for this stage.` : `${metric.requested_metric} is the primary objective for this stage.`}</strong><p>{interpretation}</p></section>
    <MetricPlot experiment={experiment} />
    <details className="experiment-details"><summary>Run details</summary><div className="experiment-details-grid"><span><b>Run</b>{experiment.id || "not recorded"}</span><span><b>Trial</b>{experiment.trial_id || "not recorded"}</span><span><b>Baseline</b>{formatText(experiment.baseline_comparison.reason || "not available")}</span><span><b>Replicates</b>{experiment.replicates.status === "legacy_single_seed" ? "legacy single-seed" : `${experiment.replicates.completed}/${experiment.replicates.required}`}</span><span><b>Lockbox</b>{experiment.lockbox.status.replaceAll("_", " ")}</span><span><b>Checkpoint</b>{experiment.checkpoint ? "available" : "not recorded"}</span></div></details>
  </Card>;
}

export function ExpectedOutcomes({ data }: { data: Investigation }) {
  const outcomes = data.expected_outcomes;
  return <Card title="Expected outcomes" eyebrow="WHAT WOULD CHANGE OUR MIND"><div className={`registration-banner registration-${outcomes.registration_status}`}><strong>{outcomes.registration_status === "missing" ? "This experiment has no registered predictions and cannot currently distinguish competing hypotheses." : outcomes.registration_status === "partial" ? "Predictions or falsification criteria are incomplete." : "Predictions and falsification criteria were registered before the experiment."}</strong></div><div className="outcomes-list">{outcomes.predictions.length ? outcomes.predictions.map((prediction) => <div className="outcome-row" key={prediction.id}><div><strong>{formatText(prediction.statement)}</strong><span className="muted">Hypotheses: {prediction.hypothesis_ids.join(", ") || "not linked"} · {prediction.source.replaceAll("_", " ")}</span></div><Badge tone={prediction.observed_status === "contradicts_prediction" ? "critical" : prediction.observed_status === "matches_prediction" ? "healthy" : prediction.observed_status === "not_yet_observed" ? "awaiting" : "neutral"}>{prediction.observed_status.replaceAll("_", " ")}</Badge></div>) : <div className="empty">No predictions are registered for this experiment.</div>}</div>{outcomes.falsification_criteria.length > 0 && <div className="falsification-list"><span className="eyebrow">FALSIFICATION CRITERIA</span>{outcomes.falsification_criteria.map((criterion) => <p key={criterion}>• {formatText(criterion)}</p>)}</div>}</Card>;
}

export function DomainGuidanceCard({ data, onSelect }: { data: Investigation; onSelect: (title: string, value: unknown) => void }) {
  const guidance = data.domain_guidance;
  if (!guidance.enabled) return <Card title="Domain guidance" eyebrow="SCIENTIFIC CONSTRAINTS"><div className="empty">No campaign-specific domain contract is configured.</div></Card>;
  const toneFor = (status: string) => lifecycleTone(status);
  const artifacts = Object.entries(guidance.artifacts || {});
  return <Card title="Domain guidance" eyebrow="BIOLOGICAL AND IDENTIFIABILITY TESTS" className="domain-guidance-card">
    <div className="domain-summary">
      <div><Badge tone={toneFor(guidance.status)}>{guidance.status.replaceAll("_", " ")}</Badge><p>Classification identifiability is a hard guardrail. Biological latent-space geometry is secondary evidence and cannot rescue a failed classifier.</p></div>
      <div className="domain-meta"><span><b>Contract</b>{guidance.contract?.id || "unavailable"}</span><span><b>Calibration</b>{guidance.calibration?.status || "required"} · {guidance.calibration?.replicate_count || 0} replicates</span><span><b>Candidate coverage</b>{guidance.replicate_coverage.completed}/{guidance.replicate_coverage.required} · seeds {guidance.replicate_coverage.seeds.join(", ") || "not registered"}</span><span><b>Evaluation unit</b>{String(guidance.unit_of_analysis.outer || "compound")} → {String(guidance.unit_of_analysis.inner || "experimental run")}</span><span><b>Projection policy</b>UMAP is visualization only</span></div>
    </div>
    <div className="domain-constraint-list">
      {guidance.constraints.map((constraint) => <section className={`domain-constraint domain-${constraint.status}`} key={constraint.id}>
        <button className="domain-constraint-heading" onClick={() => onSelect(constraint.title, constraint)}><span className="domain-constraint-heading-main"><span className="domain-constraint-status"><Badge tone={toneFor(constraint.status)}>{constraint.status.replaceAll("_", " ")}</Badge><Badge tone="neutral">{constraint.role.replaceAll("_", " ")}</Badge></span><strong>{constraint.title}</strong><small>{constraint.labels.join(" ↔ ")} · {constraint.support.sufficient === false ? "independent-unit support insufficient" : constraint.support.sufficient === true ? "independent-unit support available" : "independent-unit support not evaluated"}</small></span></button>
        <div className="domain-checks">{constraint.checks.map((check) => {
          const baseline = check.baseline as Record<string, unknown> | null | undefined;
          const interval = check.confidence_interval_95;
          return <button className={`domain-check domain-check-${check.status.replaceAll("_", "-")}`} key={`${constraint.id}-${check.metric}`} onClick={() => onSelect(check.metric.replaceAll("_", " "), check)}>
            <span className="domain-check-top"><span className="domain-metric-name">{check.metric.replaceAll("_", " ")}</span><Badge tone={toneFor(check.status)}>{check.status.replaceAll("_", " ")}</Badge></span>
            <span className="domain-check-value"><strong>{formatNumber(check.value)}</strong><span>95% CI {interval ? `${formatNumber(interval[0])}–${formatNumber(interval[1])}` : "unavailable"}</span></span>
            <span className="domain-check-meta"><span>baseline {formatNumber(typeof baseline?.mean === "number" ? baseline.mean : null)}</span><span>Δ {formatNumber(check.delta)}</span></span>
          </button>;
        })}</div>
      </section>)}
    </div>
    <div className="domain-footer"><span>Split hash: {guidance.split_hash ? guidance.split_hash.slice(0, 16) : "not evaluated"} · Contract hash: {guidance.contract?.hash ? guidance.contract.hash.slice(0, 16) : "unavailable"}</span>{artifacts.length > 0 && <button onClick={() => onSelect("Domain evaluation artifacts", guidance.artifacts)}>{artifacts.length} artifact groups</button>}</div>
  </Card>;
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
  const alternatives = data.candidates.filter((candidate) => candidate.id !== preferred?.id).slice(0, 2);
  return <Card title="Next decisions" eyebrow="AUTONOMOUS POLICY QUEUE">
    {preferred && <section className="automatic-next"><div className="decision-heading"><span className="eyebrow">NEXT AUTOMATIC DECISION</span><span className="decision-badge decision-badge-automatic">Automatic next</span></div><strong>{formatText(preferred.title)}</strong><p>{formatText(preferred.rationale || preferred.purpose || "No rationale recorded.")}</p><span className="muted">information gain {formatNumber(preferred.expected_information_gain)} · scientific value {formatNumber(preferred.scientific_value)}</span></section>}
    <section className="decision-alternatives"><div className="decision-heading"><span className="eyebrow">OTHER SELECTABLE OPTIONS</span><span className="muted">choose to inspect</span></div><div className="queue-list">{alternatives.length ? alternatives.map((candidate) => <button className="queue-item" key={candidate.id} onClick={() => onSelect(candidate)}><div><strong>{formatText(candidate.title)}</strong><span>{formatText(candidate.rationale || candidate.purpose || "No rationale recorded.")}</span><small>{candidate.status}{candidate.validation_reasons.length ? ` · ${formatText(candidate.validation_reasons.join("; "))}` : ""}</small></div><span className="decision-badge decision-badge-selectable">Selectable</span></button>) : <div className="empty">No alternative decisions are currently recorded.</div>}</div></section>
    <p className="muted queue-note">The automatic decision is shown once. Select an alternative to inspect its details.</p>
  </Card>;
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
  const relations = [...new Set(data.graph.edges.map((edge) => edge.relation).filter((relation): relation is string => Boolean(relation)))].sort();
  const relationColor = (relation: string) => data.graph.edges.find((edge) => edge.relation === relation)?.color || "#777777";
  return <Card title="Focused reasoning graph" eyebrow="SECONDARY DRILL-DOWN"><div className="graph-controls"><label>detail <input type="range" min="0" max="5" value={level} onChange={(event) => onLevel(Number(event.target.value))} /></label><label>relations <input type="range" min="0" max="5" value={relationDepth} onChange={(event) => onDepth(Number(event.target.value))} /></label><label>scale <input aria-label="Scale graph" type="range" min="40" max="100" step="5" value={scale} onChange={(event) => onScale(Number(event.target.value))} /><span>{scale}%</span></label><label>node type <select value={entityType} onChange={(event) => onEntityType(event.target.value)}><option value="">all</option>{kinds.map((kind) => <option key={kind}>{kind}</option>)}</select></label><label>relation <select value={relationType} onChange={(event) => onRelationType(event.target.value)}><option value="">all</option>{relations.map((relation) => <option key={relation}>{relation}</option>)}</select></label><span className="muted">{data.graph.nodes.length} nodes · {data.graph.edges.length} edges</span></div><div className="graph-legend"><span><i className="node-key node-hypothesis" /> hypothesis</span><span><i className="node-key node-observation" /> observation</span><span><i className="node-key node-experiment" /> experiment/stage</span><span>→ relation direction</span></div>{relations.length > 0 && <div className="graph-relation-legend"><span className="eyebrow">EDGE RELATIONS</span>{relations.map((relation) => <span className="relation-legend-item" key={relation}><i className="relation-key" style={{ borderTopColor: relationColor(relation) }} />{relation}</span>)}</div>}{data.graph.svg ? <div className="graph-svg"><div className="graph-canvas" style={{ transform: `scale(${scale / 100})`, transformOrigin: "top left" }} dangerouslySetInnerHTML={{ __html: data.graph.svg }} /></div> : <div className="empty">No focused graph evidence is available.</div>}<div className="graph-node-list"><span className="eyebrow">SELECT A NODE FOR DETAILS</span>{data.graph.nodes.map((node) => <button key={node.id} onClick={() => onNodeSelect(node)}><span className="node-picker-heading"><span className="node-key" style={{ backgroundColor: node.color || "var(--blue)" }} /><span className="node-picker-kind">{node.kind}</span></span><span className="node-picker-summary">{nodePickerSummary(node)}</span></button>)}</div></Card>;
}

function nodePickerSummary(node: { label: string; kind: string }): string {
  const text = node.label.replace(/\s+/g, " ").trim();
  if (node.kind === "trial") {
    const trial = text.match(/^TRIAL\s+(\d+)\s*/i);
    if (trial) return `#${trial[1]} ${text.slice(trial[0].length).replace(/^trial:\s*/i, "")}`.trim();
  }
  const escapedKind = node.kind.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const prefix = new RegExp(`^${escapedKind}(?::|\\s+)\\s*`, "i");
  return text.replace(prefix, "").trim() || text;
}

function PrettyString({ value }: { value: string }) {
  const lines = value.split("\n");
  const keyValuePattern = /^([A-Za-z][A-Za-z0-9 _-]{0,30}:)\s*(.*)$/;
  const hasSecondaryKeys = lines.some((line) => /^[A-Za-z][A-Za-z0-9 _-]{0,30}:\s*/.test(line));
  if (!hasSecondaryKeys) return <span className="pretty-string">{value}</span>;
  return <span className="pretty-string pretty-structured-text">{lines.map((line, index) => {
    const match = line.match(keyValuePattern);
    if (!match) return <span className="pretty-text-line" key={index}>{line}</span>;
    return <span className="pretty-text-line" key={index}><strong className="pretty-secondary-key">{match[1]}</strong><span className="pretty-text-value">{match[2]}</span></span>;
  })}</span>;
}

function PrettyValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null) return <span className="pretty-null">null</span>;
  if (typeof value === "string") return <PrettyString value={value} />;
  if (typeof value === "number") return <span className="pretty-number">{formatNumber(value)}</span>;
  if (typeof value === "boolean") return <span className="pretty-boolean">{String(value)}</span>;
  if (Array.isArray(value)) {
    return <div className="pretty-array">{value.length ? value.map((item, index) => <div className="pretty-array-item" key={index}><PrettyValue value={item} depth={depth + 1} /></div>) : <span className="pretty-null">empty list</span>}</div>;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return <div className="pretty-object">{entries.length ? entries.map(([key, item]) => <div className={`pretty-field ${depth > 0 ? "pretty-nested-field" : ""}`} key={key}><strong>{key}</strong><span className="pretty-colon">:</span><PrettyValue value={item} depth={depth + 1} /></div>) : <span className="pretty-null">empty object</span>}</div>;
  }
  return <span>{String(value)}</span>;
}

export function DetailDrawer({ detail, onClose }: { detail: { title: string; value: unknown } | null; onClose: () => void }) {
  const [rawJson, setRawJson] = useState(false);
  useEffect(() => { setRawJson(false); }, [detail]);
  if (!detail) return null;
  const raw = JSON.stringify(detail.value, null, 2) ?? String(detail.value);
  return <div className="drawer-backdrop" onClick={onClose}><aside className="drawer" onClick={(event) => event.stopPropagation()}><button className="close-button" onClick={onClose}>×</button><div className="eyebrow">DETAIL</div><h2>{detail.title}</h2><div className="detail-view-toggle" role="group" aria-label="Detail rendering"><button className={!rawJson ? "selected" : ""} onClick={() => setRawJson(false)}>Pretty</button><button className={rawJson ? "selected" : ""} onClick={() => setRawJson(true)}>Original JSON</button></div>{rawJson ? <pre>{raw}</pre> : <div className="pretty-json"><PrettyValue value={detail.value} /></div>}</aside></div>;
}
