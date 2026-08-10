import type { Game, StrategyResponse } from "../types";

interface StrategyDisplayProps {
    game: Game;
    infoset: string | null;
    strategy: StrategyResponse | null;
}

const ACTION_COLORS: Record<string, string> = {
    f: "bg-red-500",
    c: "bg-gray-400",
    k: "bg-yellow-400",
    b: "bg-green-500",
    r: "bg-emerald-500",
};

const ACTION_LABELS: Record<string, string> = {
    f: "Fold",
    c: "Check",
    k: "Call",
    b: "Bet",
    r: "Raise",
};

// Check if a Kuhn history is terminal
function kuhnIsTerminal(h: string): boolean {
    return ["cc", "bc", "bf", "cbc", "cbf"].includes(h);
}

// Check if a Leduc history is terminal
function leducIsTerminal(history: string): boolean {
    const FOLD = new Set(["bf", "brf", "cbf", "cbrf"]);
    const NOFOLD = new Set(["cc", "bk", "brk", "cbk", "cbrk"]);
    const parts = history.split("/");
    for (const seg of parts) {
        if (FOLD.has(seg)) return true;
    }
    if (parts.length === 3 && NOFOLD.has(parts[2])) return true;
    return false;
}

export default function StrategyDisplay({
    game,
    infoset,
    strategy,
}: StrategyDisplayProps) {
    if (!strategy) {
        return (
            <div className="flex h-32 items-center justify-center">
                <p className="text-gray-500">Train the model first</p>
            </div>
        );
    }

    if (!infoset) {
        return (
            <div className="flex h-32 items-center justify-center">
                <p className="text-gray-500">Select a card and build a history</p>
            </div>
        );
    }

    // Extract history from infoset key to check terminal
    const historyPart =
        game === "kuhn"
            ? infoset.split(":")[1] ?? ""
            : infoset.split(":")[1] ?? "";

    const isTerminal =
        game === "kuhn"
            ? kuhnIsTerminal(historyPart)
            : leducIsTerminal(historyPart);

    if (isTerminal) {
        return (
            <div className="flex h-32 items-center justify-center">
                <p className="text-gray-500 italic">
                    Terminal state — no actions available
                </p>
            </div>
        );
    }

    const actionProbs = strategy.strategy[infoset];

    if (!actionProbs) {
        return (
            <div className="flex h-32 items-center justify-center">
                <p className="text-gray-500">
                    Infoset <code className="text-gray-400">"{infoset}"</code> not found
                    in strategy
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                GTO Strategy at{" "}
                <code className="text-blue-400">{infoset}</code>
            </p>
            <div className="space-y-2">
                {Object.entries(actionProbs).map(([action, prob]) => (
                    <div key={action} className="flex items-center gap-3">
                        <span className="w-14 text-right text-sm text-gray-400">
                            {ACTION_LABELS[action] ?? action}
                        </span>
                        <div className="flex-1 overflow-hidden rounded-full bg-gray-800">
                            <div
                                className={`h-6 rounded-full ${ACTION_COLORS[action] ?? "bg-gray-500"} flex items-center pl-2 text-xs font-bold text-gray-900 transition-all`}
                                style={{ width: `${Math.max(prob * 100, 1)}%` }}
                            >
                                {prob > 0.05 && `${(prob * 100).toFixed(1)}%`}
                            </div>
                        </div>
                        <span className="w-14 text-sm text-gray-400">
                            {(prob * 100).toFixed(1)}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}
