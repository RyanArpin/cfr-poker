export type Game = "kuhn" | "leduc";

export interface TrainResponse {
    game: Game;
    iterations: number;
    nodes: number;
    final_ev: number;
}

export interface StrategyResponse {
    game: Game;
    strategy: Record<string, Record<string, number>>;
}

export interface InfosetResponse {
    game: Game;
    infoset: string;
    strategy: Record<string, number>;
}

export interface ExploitabilityResponse {
    game: Game;
    exploitability: number;
}

export interface EVResponse {
    game: Game;
    ev: number;
}

export interface HealthResponse {
    status: string;
    trained_games: string[];
}
