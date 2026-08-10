import { useState } from "react";
import type { Game, StrategyResponse } from "./types";
import GameSelector from "./components/GameSelector";
import TrainButton from "./components/TrainButton";
import CardSelector from "./components/CardSelector";
import HistoryBuilder from "./components/HistoryBuilder";
import StrategyDisplay from "./components/StrategyDisplay";
import StatsBar from "./components/StatsBar";

function buildInfoset(
    game: Game,
    privateCard: string | null,
    communityCard: string | null,
    history: string
): string | null {
    if (!privateCard) return null;
    if (game === "kuhn") {
        return `${privateCard}:${history}`;
    }
    // Leduc: "<card>/<community_or_->:<history>"
    const comm = communityCard ?? "-";
    return `${privateCard}/${comm}:${history}`;
}

export default function App() {
    const [game, setGame] = useState<Game>("kuhn");
    const [strategy, setStrategy] = useState<StrategyResponse | null>(null);
    const [privateCard, setPrivateCard] = useState<string | null>(null);
    const [communityCard, setCommunityCard] = useState<string | null>(null);
    const [history, setHistory] = useState("");
    const [nodes, setNodes] = useState<number | null>(null);
    const [ev, setEv] = useState<number | null>(null);
    const [exploitability, setExploitability] = useState<number | null>(null);

    function handleGameChange(g: Game) {
        setGame(g);
        setStrategy(null);
        setPrivateCard(null);
        setCommunityCard(null);
        setHistory("");
        setNodes(null);
        setEv(null);
        setExploitability(null);
    }

    function handleTrained(
        strat: StrategyResponse,
        n: number,
        e: number,
        exploit: number
    ) {
        setStrategy(strat);
        setNodes(n);
        setEv(e);
        setExploitability(exploit);
    }

    const infoset = buildInfoset(game, privateCard, communityCard, history);

    return (
        <div className="min-h-screen bg-gray-950 text-white">
            <div className="mx-auto max-w-6xl px-4 py-10">
                {/* Header */}
                <header className="mb-8 text-center">
                    <h1 className="text-4xl font-bold tracking-tight">
                        CFR Poker Solver
                    </h1>
                    <p className="mt-2 text-lg text-gray-400">
                        GTO strategy explorer — Kuhn &amp; Leduc Poker
                    </p>
                </header>

                {/* Main layout: two columns on desktop */}
                <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
                    {/* Left column — controls */}
                    <div className="space-y-6 lg:col-span-2">
                        {/* Game selector + train */}
                        <div className="flex flex-wrap items-center gap-4">
                            <GameSelector game={game} onChange={handleGameChange} />
                            <TrainButton game={game} onTrained={handleTrained} />
                        </div>

                        {/* Card selector */}
                        <div className="rounded-lg border border-gray-800 p-4">
                            <CardSelector
                                game={game}
                                privateCard={privateCard}
                                communityCard={communityCard}
                                onPrivateCard={setPrivateCard}
                                onCommunityCard={setCommunityCard}
                            />
                        </div>

                        {/* History builder */}
                        <div className="rounded-lg border border-gray-800 p-4">
                            <HistoryBuilder
                                game={game}
                                history={history}
                                onChange={setHistory}
                            />
                        </div>
                    </div>

                    {/* Right column — strategy display + stats */}
                    <div className="space-y-6 lg:col-span-3">
                        <div className="rounded-lg border border-gray-800 p-6">
                            <StrategyDisplay
                                game={game}
                                infoset={infoset}
                                strategy={strategy}
                            />
                        </div>

                        <div className="rounded-lg border border-gray-800 p-4">
                            <StatsBar
                                nodes={nodes}
                                ev={ev}
                                exploitability={exploitability}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
