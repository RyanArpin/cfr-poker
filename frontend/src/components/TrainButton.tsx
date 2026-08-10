import { useState } from "react";
import type { Game, StrategyResponse } from "../types";
import { trainGame, getStrategy, getExploitability } from "../api";

interface TrainButtonProps {
    game: Game;
    onTrained: (
        strategy: StrategyResponse,
        nodes: number,
        ev: number,
        exploitability: number
    ) => void;
}

export default function TrainButton({ game, onTrained }: TrainButtonProps) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function handleTrain() {
        setLoading(true);
        setError(null);
        try {
            const iterations = game === "kuhn" ? 10000 : 1000;
            const trainRes = await trainGame(game, iterations);
            const strategyRes = await getStrategy(game);
            const exploitRes = await getExploitability(game);
            onTrained(
                strategyRes,
                trainRes.nodes,
                trainRes.final_ev,
                exploitRes.exploitability
            );
        } catch (e) {
            setError(e instanceof Error ? e.message : "Training failed");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="flex items-center gap-3">
            <button
                onClick={handleTrain}
                disabled={loading}
                className="flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
                {loading && (
                    <svg
                        className="h-4 w-4 animate-spin"
                        viewBox="0 0 24 24"
                        fill="none"
                    >
                        <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                        />
                        <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                    </svg>
                )}
                {loading ? "Training…" : "Train"}
            </button>
            {error && <p className="text-sm text-red-400">{error}</p>}
        </div>
    );
}
