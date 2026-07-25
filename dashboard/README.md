# ZebraFish Mission Control

Mission Control is a read-only React/TypeScript dashboard for the autonomous
scientific experiment controller. It turns the YAML state and campaign
artifacts into a compact investigation story: question, hypothesis, evidence,
running experiment, expected outcomes, alerts, and next automatic decision.

## Run the built dashboard

From the repository root:

```bash
cd dashboard
npm install
npm run build
cd ..
./run_campaign dashboard cnn
```

Open <http://127.0.0.1:8000>. The API is bound to localhost and exposes only
GET endpoints. It cannot approve, reject, launch, terminate, or mutate a
campaign.

## Development

Run the API and Vite development server in separate terminals:

```bash
.venv/bin/uvicorn src.dashboard_api:app --reload
cd dashboard
npm run dev
```

The Vite server proxies `/api` requests to `127.0.0.1:8000`. The dashboard
refreshes the investigation every ten seconds. Use the campaign selector,
current/history/full view tabs, graph detail slider, and relation-depth slider
to change the read-only view.

## Data behavior

The Python adapter reads `state/scientific_state.yaml`, campaign state, stage
status, run histories, summary metrics, checkpoints, and deterministic
observations. Missing beliefs or predictions are displayed as unavailable.
Live process and fresh artifact evidence takes precedence over stale controller
metadata; stale metadata is surfaced as a warning instead of falsely showing a
running process as stopped.
