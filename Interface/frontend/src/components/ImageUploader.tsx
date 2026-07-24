import { useEffect, useState } from "react";

interface ImageUploaderProps {
  image: File | null;
  setImage: React.Dispatch<React.SetStateAction<File | null>>;
}

export default function ImageUploader({
  image,
  setImage,
}: ImageUploaderProps) {
  const [previewUrl, setPreviewUrl] = useState("");
  const [showPreview, setShowPreview] = useState(false);

  useEffect(() => {
    if (!image) {
      setPreviewUrl("");
      setShowPreview(false);
      return;
    }

    const url = URL.createObjectURL(image);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [image]);

  const handleImageChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!e.target.files?.length) return;

    setImage(e.target.files[0]);
  };

  return (
    <div className="mt-6">

      <label className="mb-2 block text-lg font-semibold text-gray-200">
        Image
      </label>

      {/* Hidden File Input */}

      <input
        id="image-upload"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="hidden"
      />

      <div className="flex flex-wrap items-center gap-3">

        {/* Choose Image */}

        <label
          htmlFor="image-upload"
          className="max-w-xs cursor-pointer rounded-lg bg-cyan-500 px-5 py-2.5 font-semibold text-white transition hover:bg-cyan-600 truncate"
          title={image?.name}
        >
          {image ? `🖼️ ${image.name}` : "📁 Choose Image"}
        </label>

        {/* Preview Button */}

        {image && (
          <button
            type="button"
            onClick={() => setShowPreview(!showPreview)}
            className="rounded-lg border border-cyan-500 px-5 py-2.5 font-semibold text-cyan-400 transition hover:bg-cyan-500 hover:text-white"
          >
            {showPreview ? "Hide Preview" : "Preview"}
          </button>
        )}
      </div>


      <div
        className={`overflow-hidden transition-all duration-300 ${
          showPreview ? "max-h-[500px] mt-5" : "max-h-0"
        }`}
      >
        {previewUrl && (
          <img
            src={previewUrl}
            alt="Preview"
            className="w-full max-h-96 rounded-xl border border-slate-700 object-contain"
          />
        )}
      </div>
    </div>
  );
}