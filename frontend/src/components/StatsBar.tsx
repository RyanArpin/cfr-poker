interface StatsBarProps {
    nodes: number | null;
    ev: number | null;
    exploitability: number | null;
}

export default function StatsBar({ nodes, ev, exploitability }: StatsBarProps) {
    if (nodes === null || ev === null || exploitability === null) {
        return null; // hidden until training completes
    }

    let exploitColor = "bg-red-500";
    if (exploitability < 0.01) exploitColor = "bg-green-500";
    else if (exploitability < 0.05) exploitColor = "bg-yellow-500";

    return (
        <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
                <p className="text-2xl font-bold text-white">{nodes.toLocaleString()}</p>
                <p className="text-xs text-gray-500 uppercase tracking-wider">Nodes</p>
            </div>
            <div className="text-center">
                <p className="text-2xl font-bold text-white">{ev.toFixed(4)}</p>
                <p className="text-xs text-gray-500 uppercase tracking-wider">
                    Game Value (P1)
                </p>
            </div>
            <div className="text-center">
                <div className="flex items-center justify-center gap-2">
                    <span className={`inline-block h-2.5 w-2.5 rounded-full ${exploitColor}`} />
                    <p className="text-2xl font-bold text-white">
                        {Math.abs(exploitability) < 0.0001
                            ? "≈ 0"
                            : exploitability.toFixed(4)}
                    </p>
                </div>
                <p className="text-xs text-gray-500 uppercase tracking-wider">
                    Exploitability
                </p>
            </div>
        </div>
    );
}
