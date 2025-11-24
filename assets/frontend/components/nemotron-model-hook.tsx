"use client"

import { useState, useEffect } from "react"
import { langChainService } from "@/lib/langchain-service"
import { HumanMessage, SystemMessage } from "@langchain/core/messages"
import { ChatOpenAI } from "@langchain/openai"

export function useNemotronModel() {
  const [model, setModel] = useState<ChatOpenAI | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    const initializeModel = async () => {
      try {
        // Get the model from the service
        const nemotronModel = await langChainService.getNemotronModel()
        setModel(nemotronModel)
        setIsLoading(false)
      } catch (err) {
        console.error("Error initializing Nemotron model:", err)
        setError(err instanceof Error ? err.message : "Failed to initialize model")
        setIsLoading(false)
      }
    }
    
    initializeModel()
  }, [])
  
  const generateResponse = async (
    userInput: string, 
    systemPrompt: string = "You are a helpful, concise assistant.",
    options?: {
      temperature?: number;
      maxTokens?: number;
    }
  ) => {
    if (!model) {
      throw new Error("Model not initialized")
    }
    
    const messages = [
      new SystemMessage(systemPrompt),
      new HumanMessage(userInput)
    ]
    
    // Option to override model settings
    if (options) {
      try {
        const customModel = await langChainService.getNemotronModel({
          temperature: options.temperature,
          maxTokens: options.maxTokens
        })
        
        return await customModel.invoke(messages)
      } catch (error) {
        console.error("Error with custom model:", error)
        throw error
      }
    }
    
    return await model.invoke(messages)
  }
  
  return {
    model,
    isLoading,
    error,
    generateResponse
  }
} 