interface PredictionCardProps {
  prediction: string;
  confidence: number;
}

export default function PredictionCard({
  prediction,
  confidence,
}: PredictionCardProps) {
  const isFake = prediction.toLowerCase() === "fake";

  return (
    <div
      className={`mt-10 rounded-2xl border p-6 shadow-lg transition-all duration-300 ${
        isFake
          ? "border-red-500 bg-red-950/40"
          : "border-green-500 bg-green-950/40"
      }`}
    >
      <h2 className="mb-6 text-center text-2xl font-bold">
        Analysis Result
      </h2>

      <div className="text-center">
        <p
          className={`text-4xl font-extrabold ${
            isFake ? "text-red-400" : "text-green-400"
          }`}
        >
          {isFake ? "🔴 FAKE NEWS" : "🟢 REAL NEWS"}
        </p>

        <p className="mt-6 text-lg text-gray-300">
          Confidence
        </p>

        <p className="mt-2 text-3xl font-bold">
          {(confidence * 100).toFixed(2)}%
        </p>

        {/* Progress Bar */}

        <div className="mt-6 h-3 w-full overflow-hidden rounded-full bg-slate-700">
          <div
            className={`h-full transition-all duration-700 ${
              isFake ? "bg-red-500" : "bg-green-500"
            }`}
            style={{
              width: `${confidence * 100}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}