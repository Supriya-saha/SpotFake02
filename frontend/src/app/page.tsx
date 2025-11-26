"use client"

import { useState } from "react"
import Image from "next/image"
import { Upload, Loader2, CheckCircle2, XCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"

const API_URL = "http://localhost:8000"

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
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-purple-700 to-indigo-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">🔍 SpotFake</h1>
          <p className="text-xl text-purple-100">AI-Powered Fake News Detection with Explainability</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Image Upload */}
          <Card>
            <CardHeader>
              <CardTitle>📸 Upload Image</CardTitle>
              <CardDescription>Upload the image associated with the news</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Label htmlFor="image-upload" className="cursor-pointer">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-purple-500 transition-colors">
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
                        <Upload className="h-12 w-12 text-gray-400" />
                        <span className="text-sm text-gray-600">Click to upload image</span>
                        <span className="text-xs text-gray-400">JPEG, PNG (max 10MB)</span>
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
                  <p className="text-sm text-gray-600">Selected: {imageFile.name}</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Text Input */}
          <Card>
            <CardHeader>
              <CardTitle>📝 Enter Text</CardTitle>
              <CardDescription>Enter the news text content to analyze</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter the news text content here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-[280px] resize-none"
              />
            </CardContent>
          </Card>
        </div>

        {/* Options & Submit */}
        <Card className="mb-8">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex flex-col sm:flex-row gap-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeGradcam}
                    onChange={(e) => setIncludeGradcam(e.target.checked)}
                    className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm">🎨 Include Grad-CAM Visualization</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeShap}
                    onChange={(e) => setIncludeShap(e.target.checked)}
                    className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm">📊 Include SHAP Explanation (~30s)</span>
                </label>
              </div>
              <Button
                onClick={handleAnalyze}
                disabled={isLoading || !imageFile || !text.trim()}
                className="bg-purple-600 hover:bg-purple-700 text-white"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Analyze News"
                )}
              </Button>
            </div>
            {isLoading && includeShap && (
              <p className="text-sm text-purple-600 mt-2 text-center">
                ⏳ SHAP analysis is running, this may take 20-40 seconds...
              </p>
            )}
          </CardContent>
        </Card>

        {/* Error Message */}
        {error && (
          <Card className="mb-8 border-red-500 bg-red-50">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-2 text-red-700">
                <XCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Verdict */}
            <Card className={result.verdict === "REAL" ? "border-green-500 bg-green-50" : "border-red-500 bg-red-50"}>
              <CardContent className="pt-6">
                <div className="text-center">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    {result.verdict === "REAL" ? (
                      <CheckCircle2 className="h-12 w-12 text-green-600" />
                    ) : (
                      <XCircle className="h-12 w-12 text-red-600" />
                    )}
                    <h2 className={`text-4xl font-bold ${result.verdict === "REAL" ? "text-green-700" : "text-red-700"}`}>
                      {result.verdict}
                    </h2>
                  </div>
                  <p className={`text-xl mb-4 ${result.verdict === "REAL" ? "text-green-600" : "text-red-600"}`}>
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                  
                  {/* AI Analysis Summary */}
                  {result.analysis && (
                    <div className="mt-4 p-4 bg-white/50 rounded-lg">
                      <p className="text-sm text-gray-700 italic">&quot;{result.analysis}&quot;</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Analysis & Reasoning */}
            {(result.reasoning && result.reasoning.length > 0) && (
              <Card>
                <CardHeader>
                  <CardTitle>🧠 Analysis Reasoning</CardTitle>
                  <CardDescription>
                    Key factors identified by the detection system
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.reasoning.map((reason, idx) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <span className="text-purple-600 font-bold mt-1">•</span>
                        <span className="text-gray-700">{reason}</span>
                      </li>
                    ))}
                  </ul>
                  
                  {/* Key Indicators */}
                  {result.key_indicators && result.key_indicators.length > 0 && (
                    <div className="mt-4 pt-4 border-t">
                      <h4 className="font-semibold text-gray-700 mb-2">Key Indicators:</h4>
                      <div className="flex flex-wrap gap-2">
                        {result.key_indicators.map((indicator, idx) => (
                          <span
                            key={idx}
                            className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm"
                          >
                            {indicator}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Grad-CAM */}
              {includeGradcam && result.gradcam_image && (
                <Card>
                  <CardHeader>
                    <CardTitle>🎨 Grad-CAM Visualization</CardTitle>
                    <CardDescription>
                      Shows which image regions influenced the model&apos;s decision
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative w-full h-96">
                      <Image
                        src={`data:image/png;base64,${result.gradcam_image}`}
                        alt="Grad-CAM Heatmap"
                        fill
                        className="object-contain rounded-lg"
                      />
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* SHAP */}
              {includeShap && result.shap_explanation && result.shap_explanation.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>📊 SHAP Token Importance</CardTitle>
                    <CardDescription>
                      Top influential words in the text (Green = pushes toward REAL, Red = pushes toward FAKE)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {result.shap_explanation.map((item, idx) => {
                        const absImportance = Math.abs(item.importance)
                        const isPositive = item.importance > 0
                        // Scale opacity based on importance magnitude
                        const opacity = Math.min(0.3 + absImportance * 100, 1)
                        
                        return (
                          <div
                            key={idx}
                            className={`px-4 py-2 rounded-full text-sm font-medium transition-all hover:scale-105 ${
                              isPositive
                                ? "bg-green-500 text-white border-2 border-green-600"
                                : "bg-red-500 text-white border-2 border-red-600"
                            }`}
                            style={{ opacity }}
                            title={`Importance: ${item.importance.toFixed(6)}\n${isPositive ? 'Pushes toward REAL' : 'Pushes toward FAKE'}`}
                          >
                            {item.token} <span className="font-mono text-xs">({item.importance.toFixed(4)})</span>
                          </div>
                        )
                      })}
                    </div>
                    {result.shap_explanation.length === 0 && (
                      <p className="text-sm text-gray-500">No SHAP values available</p>
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

