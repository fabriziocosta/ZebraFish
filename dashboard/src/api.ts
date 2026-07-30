import { InvestigationSchema, type Investigation } from "./model";

export async function fetchInvestigation(
  campaign: string,
  options: { view: string; level: number; relationDepth: number; entityType?: string; relationType?: string },
): Promise<Investigation> {
  const query = new URLSearchParams({
    view: options.view,
    level: String(options.level),
    relation_depth: String(options.relationDepth),
  });
  if (options.entityType) query.set("entity_type", options.entityType);
  if (options.relationType) query.set("relation_type", options.relationType);
  const response = await fetch(`/api/investigation/${encodeURIComponent(campaign)}?${query}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Dashboard API returned ${response.status}: ${detail}`);
  }
  return InvestigationSchema.parse(await response.json());
}

export async function controlController(
  campaign: string,
  controller: "meta" | "campaign",
  action: "start" | "stop" | "continue",
): Promise<{ status: string; [key: string]: unknown }> {
  const response = await fetch(`/api/investigation/${encodeURIComponent(campaign)}/control/${controller}/${action}`, { method: "POST" });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Controller action returned ${response.status}: ${detail}`);
  }
  return response.json();
}
