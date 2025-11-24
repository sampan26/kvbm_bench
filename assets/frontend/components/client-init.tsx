"use client"

import { useEffect } from "react"
import { syncSettingsWithServer, initializeDefaultSettings } from "@/lib/client-init"

export function ClientInitializer() {
  useEffect(() => {
    // Initialize client settings
    const initSettings = async () => {
      try {
        // Set default values first
        initializeDefaultSettings()
        // Then sync with server
        await syncSettingsWithServer()
      } catch (error) {
        console.error("Failed to initialize client settings:", error)
      }
    }
    
    initSettings()
  }, [])
  
  // This component doesn't render anything
  return null
} 