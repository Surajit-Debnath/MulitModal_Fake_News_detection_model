interface NewsInputProps {
  text: string;
  setText: React.Dispatch<React.SetStateAction<string>>;
}

export default function NewsInput({
  text,
  setText,
}: NewsInputProps) {
  return (
    <div className="mb-8">
      <label
        htmlFor="news-text"
        className="mb-2 block text-lg font-semibold text-gray-200"
      >
        News Text
      </label>

      <textarea
        id="news-text"
        rows={2}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste the news article here..."
         className="w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-white resize-none outline-none transition-all focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500"
      />      
    </div>
  );
}