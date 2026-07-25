import { describe, expect, it } from "vitest";
import { InvestigationSchema } from "./model";

describe("dashboard response schema", () => {
  it("accepts the normalized live campaign shape when the API is available", async () => {
    const response = await fetch("http://127.0.0.1:8766/api/investigation/cnn").catch(() => null);
    if (!response) return;
    expect(InvestigationSchema.safeParse(await response.json()).success).toBe(true);
  });

  it("rejects malformed normalized data", () => {
    expect(InvestigationSchema.safeParse({ schema_version: "wrong" }).success).toBe(false);
  });
});
