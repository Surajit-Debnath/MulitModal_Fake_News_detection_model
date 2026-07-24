interface PredictButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export default function PredictButton({
  onClick,
  disabled = false,
}: PredictButtonProps) {
  return (
    <div className="mt-10 flex justify-center">
      <button
        onClick={onClick}
        disabled={disabled}
        className="w-full max-w-md rounded-xl bg-cyan-500 px-6 py-4 text-lg font-bold text-white transition-all duration-200 hover:bg-cyan-600 hover:scale-[1.02] active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-700"
      >
        🔍 Analyze News
      </button>
    </div>
  );
}