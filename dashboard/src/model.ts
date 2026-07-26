import { z } from "zod";

const Numeric = z.number().finite().nullable().optional();
const AnyRecord = z.record(z.string(), z.unknown());

export const EvidenceSchema = z.object({
  id: z.string(),
  type: z.string(),
  summary: z.string().default(""),
  statement: z.string().default(""),
  direction: z.enum(["supporting", "contradicting", "inconclusive", "unclassified"]),
  classification_source: z.string().default("unavailable"),
  explanation: z.string().default(""),
  source_experiments: z.array(z.string()).default([]),
  reliability: Numeric,
  evidence_strength: Numeric,
  confidence: Numeric,
  created_at: z.string().nullable().optional(),
  measurements: AnyRecord.default({}),
  detection: AnyRecord.default({}),
  references: z.array(AnyRecord).default([]),
});

export const CandidateSchema = z.object({
  id: z.string(),
  title: z.string().default("Untitled candidate"),
  purpose: z.string().default(""),
  status: z.string().default("proposed"),
  question_id: z.string().nullable().optional(),
  hypothesis_ids: z.array(z.string()).default([]),
  rationale: z.string().default(""),
  estimated_gpu_hours: Numeric,
  estimated_wall_hours: Numeric,
  expected_information_gain: Numeric,
  expected_metric_improvement: Numeric,
  scientific_value: Numeric,
  value_per_gpu_hour: Numeric,
  fixed_variables: AnyRecord.default({}),
  configuration_patch: AnyRecord.default({}),
  expected_outcomes: z.array(z.string()).default([]),
  falsification_criteria: z.array(z.string()).default([]),
  risks: z.array(z.string()).default([]),
  validation_reasons: z.array(z.string()).default([]),
  created_at: z.string().nullable().optional(),
});

export const AlertSchema = z.object({
  id: z.string(),
  severity: z.enum(["info", "warning", "critical"]),
  type: z.string(),
  condition: z.string(),
  measurements: AnyRecord.default({}),
  recommended_action: z.string(),
  automatic: z.boolean(),
});

export const GraphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  tooltip: z.string().optional(),
  kind: z.string(),
  status: z.string().optional(),
  color: z.string().optional(),
});

export const GraphSchema = z.object({
  nodes: z.array(GraphNodeSchema),
  edges: z.array(z.object({ source: z.string(), target: z.string(), relation: z.string().optional(), color: z.string().optional() })),
  svg: z.string().nullable().optional(),
});

export const MetricPointSchema = z.object({
  step: z.number(),
  epoch: z.number(),
  primary: Numeric,
  displayed: Numeric,
  train: Numeric,
  validation: Numeric,
});

export const MetricPlotSchema = z.object({
  y_axis_label: z.string().nullable().optional(),
  x_axis_label: z.string().default("epoch"),
  direction: z.string().default("unknown"),
  y_min: Numeric,
  y_max: Numeric,
  statistics: z.object({
    latest: Numeric,
    best: Numeric,
    slope: Numeric,
    train_validation_gap: Numeric,
    epochs: z.number().default(0),
  }).default({ epochs: 0 }),
  events: z.array(z.object({ id: z.string(), epoch: z.number(), type: z.string(), label: z.string(), value: Numeric })).default([]),
  comparisons: z.array(z.object({ id: z.string(), trial_id: z.string().nullable().optional(), stage: z.string(), label: z.string(), metric: z.string().nullable().optional(), points: z.array(MetricPointSchema) })).default([]),
  interpretation: z.string().default(""),
});

export const InvestigationSchema = z.object({
  schema_version: z.number(),
  campaign: z.object({ name: z.string(), id: z.string(), config_path: z.string() }),
  project: z.object({
    id: z.string(),
    name: z.string(),
    objective: z.unknown(),
    primary_metric: z.string(),
    guardrails: z.unknown(),
    remaining_gpu_hours: Numeric,
    trial_budget: z.number().nullable().optional(),
  }),
  investigation: z.object({
    status: z.string(),
    health: z.enum(["healthy", "warning", "critical"]),
    last_updated: z.string().nullable().optional(),
    view: z.string(),
  }),
  active_question: z.record(z.string(), z.unknown()).nullable(),
  active_hypothesis: z.record(z.string(), z.unknown()).nullable(),
  current_experiment: z.object({
    id: z.string().nullable().optional(),
    title: z.string(),
    status: z.string(),
    campaign_status: z.string().nullable().optional(),
    trial_id: z.string().nullable().optional(),
    stage: z.string().nullable().optional(),
    purpose: z.string().default(""),
    started_at: z.string().nullable().optional(),
    elapsed_seconds: Numeric,
    estimated_remaining_seconds: Numeric,
    progress_fraction: Numeric,
    current_epoch: z.number().nullable().optional(),
    total_epochs: z.number().nullable().optional(),
    pid: z.number().nullable().optional(),
    process_running: z.boolean(),
    run_dir: z.string().nullable().optional(),
    checkpoint: z.string().nullable().optional(),
    history_path: z.string().nullable().optional(),
    primary_metric: z.string(),
    metric_display: z.object({
      requested_metric: z.string(),
      display_metric: z.string().nullable().optional(),
      role: z.enum(["primary", "diagnostic", "unavailable"]),
      available: z.boolean(),
      direction: z.string(),
      source: z.string().nullable().optional(),
      source_path: z.string().nullable().optional(),
      fallback_reason: z.string().nullable().optional(),
    }),
    current_metric: Numeric,
    best_metric: Numeric,
    summary_metrics: z.record(z.string(), z.number()).default({}),
    metric_series: z.array(MetricPointSchema).default([]),
    metric_plot: MetricPlotSchema.default({ x_axis_label: "epoch", direction: "unknown", statistics: { epochs: 0 }, events: [], comparisons: [], interpretation: "" }),
    artifact_updated_at: z.string().nullable().optional(),
    artifact_freshness: z.record(z.string(), z.unknown()).default({}),
    compute: z.object({ consumed_gpu_hours: Numeric, expected_gpu_hours: Numeric, remaining_gpu_hours: Numeric, available: z.boolean() }),
    baseline_comparison: z.record(z.string(), z.unknown()).default({}),
  }),
  expected_outcomes: z.object({
    registration_status: z.enum(["registered", "partial", "missing"]),
    source_candidate_id: z.string().nullable().optional(),
    predictions: z.array(z.object({ id: z.string(), statement: z.string(), hypothesis_ids: z.array(z.string().nullable()).default([]), observed_status: z.string(), source: z.string() })).default([]),
    falsification_criteria: z.array(z.string()).default([]),
    missing_reason: z.string().nullable().optional(),
  }),
  evidence: z.object({ supporting: z.array(EvidenceSchema), contradicting: z.array(EvidenceSchema), inconclusive: z.array(EvidenceSchema), unclassified: z.array(EvidenceSchema), total: z.number(), counts: z.record(z.string(), z.number()).default({}) }),
  candidates: z.array(CandidateSchema).default([]),
  belief_history: z.array(z.record(z.string(), z.unknown())).default([]),
  alerts: z.array(AlertSchema).default([]),
  health: z.object({ status: z.string(), process_live: z.boolean(), artifact_freshness: z.record(z.string(), z.unknown()), controller_metadata: z.string(), intervention_required: z.boolean() }),
  controller: AnyRecord.default({}),
  graph: GraphSchema,
  diagnostics: z.object({ errors: z.array(z.string()).default([]), missing_fields: z.array(z.string()).default([]), reference_warnings: z.array(z.string()).default([]), data_coverage: z.record(z.string(), z.unknown()).default({}) }),
});

export type Investigation = z.infer<typeof InvestigationSchema>;
export type Evidence = z.infer<typeof EvidenceSchema>;
export type Candidate = z.infer<typeof CandidateSchema>;
export type Alert = z.infer<typeof AlertSchema>;
export type MetricPoint = z.infer<typeof MetricPointSchema>;
