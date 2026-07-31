import { describe, expect, it } from "vitest";
import { DomainGuidanceSchema, InvestigationSchema } from "./model";

describe("dashboard response schema", () => {
  it("accepts the normalized live campaign shape when the API is available", async () => {
    const response = await fetch("http://127.0.0.1:8766/api/investigation/cnn").catch(() => null);
    if (!response) return;
    expect(InvestigationSchema.safeParse(await response.json()).success).toBe(true);
  });

  it("rejects malformed normalized data", () => {
    expect(InvestigationSchema.safeParse({ schema_version: "wrong" }).success).toBe(false);
  });

  it("accepts uncertainty-aware domain constraints and rejects invalid roles", () => {
    const payload = {
      enabled: true,
      status: "evaluated",
      objective_eligibility: "eligible",
      contract: { id: "cnn_action_domain_v1", hash: "abc", path: "contract.yaml" },
      calibration: { status: "frozen", replicate_count: 3 },
      constraints: [{
        id: "ache_machr_separability",
        title: "AChE and mAChR remain locally distinguishable",
        role: "hard_guardrail",
        labels: ["AChE", "mAChR"],
        status: "pass",
        checks: [{
          metric: "pairwise_balanced_accuracy",
          status: "pass",
          value: 0.8,
          confidence_interval_95: [0.7, 0.9],
        }],
      }],
      unit_of_analysis: { outer: "compound", inner: "experimental_run_id" },
      sample_coverage: {},
      artifacts: {},
      warnings: [],
      umap_decision_role: "visualization_only",
    };
    expect(DomainGuidanceSchema.safeParse(payload).success).toBe(true);
    expect(DomainGuidanceSchema.safeParse({
      ...payload,
      constraints: [{ ...payload.constraints[0], role: "plot_opinion" }],
    }).success).toBe(false);
  });
});
