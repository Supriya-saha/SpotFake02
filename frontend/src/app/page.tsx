"use client"

import { useState } from "react"
import Image from "next/image"
import { Upload, Loader2, CheckCircle2, XCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"

const API_URL = "https://unhawked-jamarion-noncleistogamous.ngrok-free.dev"

interface PredictionResult {
  verdict: "REAL" | "FAKE"
  confidence: number
  raw_score: number
  text: string
  saved_image: string
  analysis?: string
  reasoning?: string[]
  key_indicators?: string[]
  gradcam_image?: string
  shap_explanation?: Array<{
    token: string
    importance: number
  }>
}

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string>("")
  const [text, setText] = useState("")
  const [includeGradcam, setIncludeGradcam] = useState(true)
  const [includeShap, setIncludeShap] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string>("")

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleAnalyze = async () => {
    if (!imageFile || !text.trim()) {
      setError("Please provide both image and text")
      return
    }

    setIsLoading(true)
    setError("")
    setResult(null)

    try {
      const formData = new FormData()
      formData.append("image", imageFile)
      formData.append("text", text)

      const url = `${API_URL}/predict?include_gradcam=${includeGradcam}&include_shap=${includeShap}`
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Prediction failed")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0e1a] relative">
      {/* Animated Grid Background */}
      <div className="grid-background"></div>
      
      <div className="container mx-auto px-4 py-8 max-w-7xl relative z-10">
        {/* Header */}
        <div className="text-center mb-12 relative">
          {/* Animated dots background */}
          <div className="dots-background">
            {[...Array(50)].map((_, i) => (
              <div key={i} className="dot"></div>
            ))}
          </div>
          
          <div className="flex items-center justify-center mb-4 relative z-10">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-lg mr-4 flex items-center justify-center">
              <span className="text-2xl font-bold text-white">F</span>
            </div>
            <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
              Fake News Detector
            </h1>
          </div>
          <p className="text-xl text-gray-100 relative z-10">Advanced Multimodal Fake News Detection</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Image Upload */}
          <Card className="bg-[#151b2e] border-gray-800">
            <CardHeader>
              <CardTitle className="text-cyan-400">Upload Image</CardTitle>
              <CardDescription className="text-gray-100">Upload the image associated with the news</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Label htmlFor="image-upload" className="cursor-pointer">
                  <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center hover:border-cyan-500 transition-colors bg-[#0a0e1a]">
                    {imagePreview ? (
                      <div className="relative w-full h-64">
                        <Image
                          src={imagePreview}
                          alt="Preview"
                          fill
                          className="object-contain rounded-lg"
                        />
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center space-y-2">
                        <Upload className="h-12 w-12 text-gray-600" />
                        <span className="text-sm text-gray-100">Click to upload image</span>
                        <span className="text-xs text-gray-600">JPEG, PNG (max 10MB)</span>
                      </div>
                    )}
                  </div>
                  <input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageChange}
                  />
                </Label>
                {imageFile && (
                  <p className="text-sm text-gray-100">Selected: {imageFile.name}</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Text Input */}
          <Card className="bg-[#151b2e] border-gray-800">
            <CardHeader>
              <CardTitle className="text-cyan-400">Enter Text</CardTitle>
              <CardDescription className="text-gray-100">Enter the news text content to analyze</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter the news text content here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-[280px] resize-none bg-[#0a0e1a] border-gray-700 text-gray-200 placeholder:text-gray-600"
              />
            </CardContent>
          </Card>
        </div>

        {/* Options & Submit */}
        <Card className="mb-8 bg-[#151b2e] border-gray-800">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex flex-col sm:flex-row gap-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeGradcam}
                    onChange={(e) => setIncludeGradcam(e.target.checked)}
                    className="w-4 h-4 text-cyan-500 rounded focus:ring-cyan-500 bg-gray-800 border-gray-600"
                  />
                  <span className="text-sm text-gray-300">Visual Context</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeShap}
                    onChange={(e) => setIncludeShap(e.target.checked)}
                    className="w-4 h-4 text-cyan-500 rounded focus:ring-cyan-500 bg-gray-800 border-gray-600"
                  />
                  <span className="text-sm text-gray-300">Relevant Text Texture</span>
                </label>
              </div>
              <Button
                onClick={handleAnalyze}
                disabled={isLoading || !imageFile || !text.trim()}
                className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white font-semibold button-glow button-pulse"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Start New Analysis"
                )}
              </Button>
            </div>
            {isLoading && includeShap && (
              <p className="text-sm text-cyan-400 mt-2 text-center">
                Deep analysis in progress, this may take 20-40 seconds...
              </p>
            )}
          </CardContent>
        </Card>

        {/* Error Message */}
        {error && (
          <Card className="mb-8 border-red-500 bg-red-950/30">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-2 text-red-400">
                <XCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Verdict and Confidence - Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Verdict Card */}
              <Card className="bg-[#151b2e] border-gray-800">
                <CardContent className="pt-6">
                  <div className="flex items-start space-x-4">
                    <div className={`w-16 h-16 rounded-full ${result.verdict === "REAL" ? "bg-green-500/20" : "bg-red-500/20"} flex items-center justify-center flex-shrink-0`}>
                      {result.verdict === "REAL" ? (
                        <CheckCircle2 className="h-10 w-10 text-green-400" />
                      ) : (
                        <XCircle className="h-10 w-10 text-red-400" />
                      )}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-gray-100 mb-1">Verdict</p>
                      <h2 className={`text-3xl font-bold ${result.verdict === "REAL" ? "text-green-400" : "text-red-400"}`}>
                        {result.verdict === "REAL" ? "REAL NEWS" : "FAKE NEWS"}
                      </h2>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Confidence Score Card */}
              <Card className="bg-[#151b2e] border-gray-800">
                <CardContent className="pt-6">
                  <div className="flex items-center space-x-4">
                    {/* Circular Progress */}
                    <div className="relative w-20 h-20 flex-shrink-0">
                      <svg className="w-20 h-20 transform -rotate-90">
                        {/* Background circle */}
                        <circle
                          cx="40"
                          cy="40"
                          r="32"
                          stroke="#1e293b"
                          strokeWidth="6"
                          fill="none"
                        />
                        {/* Progress circle */}
                        <circle
                          cx="40"
                          cy="40"
                          r="32"
                          stroke="#22d3ee"
                          strokeWidth="6"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 32}`}
                          strokeDashoffset={`${2 * Math.PI * 32 * (1 - result.confidence)}`}
                          strokeLinecap="round"
                          className="transition-all duration-1000 ease-out"
                        />
                      </svg>
                      {/* Percentage in center */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xl font-bold text-cyan-400">
                          {Math.round(result.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex-1">
                      <p className="text-sm text-gray-100 mb-1">Confidence Score</p>
                      <h3 className="text-3xl font-bold text-cyan-400">
                        {result.confidence >= 0.8 ? "High" : result.confidence >= 0.5 ? "Medium" : "Low"}
                      </h3>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* AI Analysis Summary */}
            {result.analysis && (
              <Card className="bg-[#151b2e] border-gray-800">
                <CardContent className="pt-6">
                  <p className="text-gray-300 italic leading-relaxed">&quot;{result.analysis}&quot;</p>
                </CardContent>
              </Card>
            )}

            {/* Analysis & Reasoning */}
            {(result.reasoning && result.reasoning.length > 0) && (
              <Card className="bg-[#151b2e] border-gray-800">
                <CardHeader>
                  <CardTitle className="text-cyan-400 text-2xl">Why This Result?</CardTitle>
                  <CardDescription className="text-gray-100">
                    Our AI model has identified these indicators
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="bg-[#0a0e1a] p-6 rounded-lg border border-gray-800">
                      <h3 className="text-lg font-semibold text-cyan-300 mb-4">Specific Analysis</h3>
                      <ul className="space-y-3">
                        {result.reasoning.map((reason, idx) => (
                          <li key={idx} className="flex items-start space-x-3">
                            <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <span className="text-cyan-400 text-xs font-bold">{idx + 1}</span>
                            </div>
                            <span className="text-gray-300 leading-relaxed">{reason}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  
                    {/* Key Indicators */}
                    {result.key_indicators && result.key_indicators.length > 0 && (
                      <div className="bg-[#0a0e1a] p-6 rounded-lg border border-gray-800">
                        <h4 className="font-semibold text-cyan-300 mb-3">Top Contributing Factors:</h4>
                        <div className="flex flex-wrap gap-3">
                          {result.key_indicators.map((indicator, idx) => (
                            <span
                              key={idx}
                              className="px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 rounded-lg text-sm border border-cyan-500/30 font-medium"
                            >
                              {indicator}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Grad-CAM */}
              {includeGradcam && result.gradcam_image && (
                <Card className="bg-[#151b2e] border-gray-800">
                  <CardHeader>
                    <CardTitle className="text-cyan-400">Relevant Visual Contexts</CardTitle>
                    <CardDescription className="text-gray-100">
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative w-full h-96 bg-[#0a0e1a] rounded-lg p-4 border border-gray-800">
                      <Image
                        src={`data:image/png;base64,${result.gradcam_image}`}
                        alt="Grad-CAM Heatmap"
                        fill
                        className="object-contain rounded-lg"
                      />
                    </div>
                    <div className="mt-4 text-center">
                      <p className="text-sm text-gray-100 italic">Original Image</p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* SHAP */}
              {includeShap && result.shap_explanation && result.shap_explanation.length > 0 && (
                <Card className="bg-[#151b2e] border-gray-800">
                  <CardHeader>
                    <CardTitle className="text-cyan-400">Relevant Text Texture</CardTitle>
                    <CardDescription className="text-gray-100">
                                          </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-[#0a0e1a] p-4 rounded-lg border border-gray-800">
                      <div className="flex flex-wrap gap-2">
                        {result.shap_explanation.map((item, idx) => {
                          const absImportance = Math.abs(item.importance)
                          const isPositive = item.importance > 0
                          const opacity = Math.min(0.4 + absImportance * 100, 1)
                          
                          return (
                            <div
                              key={idx}
                              className={`px-4 py-2 rounded-md text-sm font-medium transition-all hover:scale-105 ${
                                isPositive
                                  ? "bg-green-500/30 text-green-300 border border-green-500/50"
                                  : "bg-red-500/30 text-red-300 border border-red-500/50"
                              }`}
                              style={{ opacity }}
                              title={`Importance: ${item.importance.toFixed(6)}`}
                            >
                              {item.token} <span className="font-mono text-xs opacity-70">({item.importance.toFixed(4)})</span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                    {result.shap_explanation.length === 0 && (
                      <p className="text-sm text-gray-100">No significant patterns detected</p>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
            
          </div>
        )}
      </div>
    </div>
  )
}

