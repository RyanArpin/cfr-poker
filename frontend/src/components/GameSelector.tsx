import type { Game } from "../types";

interface GameSelectorProps {
    game: Game;
    onChange: (g: Game) => void;
}

export default function GameSelector({ game, onChange }: GameSelectorProps) {
    const games: { id: Game; label: string }[] = [
        { id: "kuhn", label: "Kuhn Poker" },
        { id: "leduc", label: "Leduc Poker" },
    ];

    return (
        <div className="flex gap-3">
            {games.map((g) => (
                <button
                    key={g.id}
                    onClick={() => onChange(g.id)}
                    className={`rounded-lg px-5 py-2.5 text-sm font-medium transition-all ${game === g.id
                            ? "border-2 border-blue-500 bg-gray-800 text-white"
                            : "border-2 border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600 hover:text-gray-200"
                        }`}
                >
                    {g.label}
                </button>
            ))}
        </div>
    );
}
