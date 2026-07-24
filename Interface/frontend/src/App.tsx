import { useState } from "react";

import Header from "./components/Header";
import NewsInput from "./components/NewsInput";
import ImageUploader from "./components/ImageUploader";
import PredictButton from "./components/PredictButton";
import { predictNews } from "./services/api";
import PredictionCard from "./components/Predictioncard";

function App() {
  const [text, setText] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [loading, setLoading] = useState(false);
  const  handleAnalyze = async() => {
      if (!text.trim()) {
        alert("Please enter the news text.");
        return;
      }

      if (!image) {
        alert("Please choose an image.");
        return;
      }

      
  try {
    setLoading(true);

    const result = await predictNews(text, image);

    setPrediction(result.prediction);
    setConfidence(result.confidence);
  } catch (error) {
    console.error(error);
    alert("Prediction failed.");
  } finally {
    setLoading(false);
  }
    };
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-4xl px-6 py-10">

        <Header />

        <NewsInput
          text={text}
          setText={setText}
        />

        <ImageUploader
          image={image}
          setImage={setImage}
        />
       
        <PredictButton
          onClick={handleAnalyze}
          disabled={loading}
        />

        {prediction && (
          <PredictionCard
            prediction={prediction}
            confidence={confidence}
          />
        )}
      </div>
    </div>
  );
}

export default App;