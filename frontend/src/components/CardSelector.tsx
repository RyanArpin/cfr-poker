import type { Game } from "../types";

interface CardSelectorProps {
    game: Game;
    privateCard: string | null;
    communityCard: string | null;
    onPrivateCard: (c: string) => void;
    onCommunityCard: (c: string | null) => void;
}

const CARDS = ["J", "Q", "K"];

function Card({
    label,
    selected,
    onClick,
}: {
    label: string;
    selected: boolean;
    onClick: () => void;
}) {
    return (
        <button
            onClick={onClick}
            className={`flex h-20 w-14 items-center justify-center rounded-lg border-2 text-2xl font-bold transition-all ${selected
                    ? "border-blue-500 bg-white text-gray-900 shadow-lg shadow-blue-500/20"
                    : "border-gray-600 bg-gray-100 text-gray-700 hover:border-gray-400"
                }`}
        >
            {label}
        </button>
    );
}

export default function CardSelector({
    game,
    privateCard,
    communityCard,
    onPrivateCard,
    onCommunityCard,
}: CardSelectorProps) {
    return (
        <div className="space-y-4">
            {/* Private card */}
            <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
                    Your Card
                </p>
                <div className="flex gap-3">
                    {CARDS.map((c) => (
                        <Card
                            key={c}
                            label={c}
                            selected={privateCard === c}
                            onClick={() => onPrivateCard(c)}
                        />
                    ))}
                </div>
            </div>

            {/* Community card (Leduc only) */}
            {game === "leduc" && (
                <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
                        Community Card
                    </p>
                    <div className="flex gap-3">
                        {CARDS.map((c) => (
                            <Card
                                key={c}
                                label={c}
                                selected={communityCard === c}
                                onClick={() => onCommunityCard(c)}
                            />
                        ))}
                        <button
                            onClick={() => onCommunityCard(null)}
                            className={`flex h-20 w-14 items-center justify-center rounded-lg border-2 text-xs font-medium transition-all ${communityCard === null
                                    ? "border-blue-500 bg-gray-800 text-blue-300"
                                    : "border-gray-600 bg-gray-800 text-gray-500 hover:border-gray-400"
                                }`}
                        >
                            None
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
