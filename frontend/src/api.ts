import type {
    Game,
    TrainResponse,
    StrategyResponse,
    InfosetResponse,
    ExploitabilityResponse,
    HealthResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_URL;

export async function trainGame(
    game: Game,
    iterations: number
): Promise<TrainResponse> {
    const res = await fetch(`${BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game, iterations }),
    });
    if (!res.ok) throw new Error(`Train failed: ${res.status}`);
    return res.json();
}

export async function getStrategy(game: Game): Promise<StrategyResponse> {
    const res = await fetch(`${BASE}/strategy/${game}`);
    if (!res.ok) throw new Error(`Strategy fetch failed: ${res.status}`);
    return res.json();
}

export async function getInfosetStrategy(
    game: Game,
    infoset: string
): Promise<InfosetResponse> {
    const encoded = encodeURIComponent(infoset);
    const res = await fetch(`${BASE}/strategy/${game}/${encoded}`);
    if (!res.ok) throw new Error(`Infoset fetch failed: ${res.status}`);
    return res.json();
}

export async function getExploitability(
    game: Game
): Promise<ExploitabilityResponse> {
    const res = await fetch(`${BASE}/exploitability/${game}`);
    if (!res.ok) throw new Error(`Exploitability fetch failed: ${res.status}`);
    return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
}
