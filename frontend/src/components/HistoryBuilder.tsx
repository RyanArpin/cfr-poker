import type { Game } from "../types";

interface HistoryBuilderProps {
    game: Game;
    history: string;
    onChange: (h: string) => void;
}

// ─── Kuhn legal actions ──────────────────────────────────────────────────────

function kuhnLegalActions(history: string): string[] | null {
    const terminals = new Set(["cc", "bc", "bf", "cbc", "cbf"]);
    if (terminals.has(history)) return null; // terminal
    if (history.length > 0 && history[history.length - 1] === "b") {
        return ["f", "c"]; // fold or call
    }
    return ["c", "b"]; // check or bet
}

// ─── Leduc legal actions ─────────────────────────────────────────────────────

const LEDUC_CLOSED_NO_FOLD = new Set(["cc", "bk", "brk", "cbk", "cbrk"]);
const LEDUC_CLOSED_FOLD = new Set(["bf", "brf", "cbf", "cbrf"]);

function leducRoundActions(history: string): string {
    const parts = history.split("/");
    return parts[parts.length - 1];
}

function leducIsTerminal(history: string): boolean {
    const parts = history.split("/");
    // Check for fold in any segment
    for (const seg of parts) {
        if (LEDUC_CLOSED_FOLD.has(seg)) return true;
    }
    // R2 showdown
    if (parts.length === 3 && LEDUC_CLOSED_NO_FOLD.has(parts[2])) return true;
    return false;
}

function leducIsChanceNode(history: string): boolean {
    const parts = history.split("/");
    if (parts.length === 1 && LEDUC_CLOSED_NO_FOLD.has(parts[0])) return true;
    return false;
}

function leducLegalActions(history: string): string[] | null {
    if (leducIsTerminal(history)) return null;
    if (leducIsChanceNode(history)) return null; // need community card first

    const ra = leducRoundActions(history);
    if (ra.length > 0 && (ra[ra.length - 1] === "b" || ra[ra.length - 1] === "r")) {
        const actions = ["f", "k"]; // fold, call
        if (ra.split("").filter((c) => c === "r").length < 1) {
            actions.push("r"); // raise (max 1 per round)
        }
        return actions;
    }
    return ["c", "b"]; // check or bet
}

// ─── Action display names and styling ────────────────────────────────────────

const ACTION_META: Record<string, { label: string; color: string }> = {
    c: { label: "Check", color: "bg-gray-600 hover:bg-gray-500" },
    b: { label: "Bet", color: "bg-green-600 hover:bg-green-500" },
    f: { label: "Fold", color: "bg-red-600 hover:bg-red-500" },
    k: { label: "Call", color: "bg-yellow-600 hover:bg-yellow-500" },
    r: { label: "Raise", color: "bg-emerald-600 hover:bg-emerald-500" },
};

function historyToReadable(history: string, game: Game): string {
    if (!history) return "(start)";
    if (game === "kuhn") {
        return history
            .split("")
            .map((ch) => ACTION_META[ch]?.label ?? ch)
            .join(" → ");
    }
    // Leduc
    return history
        .split("/")
        .map((seg, i) => {
            if (i === 1) return `| ${seg} |`; // community card
            return seg
                .split("")
                .map((ch) => ACTION_META[ch]?.label ?? ch)
                .join(" → ");
        })
        .join(" ");
}

export default function HistoryBuilder({
    game,
    history,
    onChange,
}: HistoryBuilderProps) {
    const actions =
        game === "kuhn" ? kuhnLegalActions(history) : leducLegalActions(history);

    const isChance = game === "leduc" && leducIsChanceNode(history);
    const isTerminal = actions === null && !isChance;

    function handleAction(action: string) {
        onChange(history + action);
    }

    function handleReset() {
        onChange("");
    }

    // For Leduc: after R1 closes, add "/" + community card + "/" to start R2
    function handleCommunityTransition(card: string) {
        onChange(history + "/" + card + "/");
    }

    return (
        <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                Action History
            </p>

            {/* Breadcrumb */}
            <p className="min-h-[1.5rem] text-sm text-gray-300">
                {historyToReadable(history, game)}
            </p>

            {/* Action buttons */}
            <div className="flex flex-wrap gap-2">
                {isTerminal && (
                    <p className="text-sm italic text-gray-500">
                        Terminal state — hand is over
                    </p>
                )}
                {isChance && (
                    <div className="space-y-1">
                        <p className="text-xs text-gray-400">
                            Select community card to continue:
                        </p>
                        <div className="flex gap-2">
                            {["J", "Q", "K"].map((c) => (
                                <button
                                    key={c}
                                    onClick={() => handleCommunityTransition(c)}
                                    className="rounded bg-purple-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-purple-600"
                                >
                                    {c}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
                {actions &&
                    actions.map((a) => {
                        const meta = ACTION_META[a];
                        return (
                            <button
                                key={a}
                                onClick={() => handleAction(a)}
                                className={`rounded px-4 py-1.5 text-sm font-medium text-white transition-colors ${meta.color}`}
                            >
                                {meta.label}
                            </button>
                        );
                    })}
            </div>

            {/* Reset */}
            <button
                onClick={handleReset}
                className="text-xs text-gray-500 underline hover:text-gray-300"
            >
                Reset history
            </button>
        </div>
    );
}
