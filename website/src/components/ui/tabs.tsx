"use client"

import * as React from "react"

interface TabsProps {
  value?: string
  onValueChange?: (value: string) => void
  children: React.ReactNode
  className?: string
}

const TabsContext = React.createContext<{
  value: string
  onValueChange: (value: string) => void
}>({ value: "", onValueChange: () => {} })

function Tabs({ value: controlledValue, onValueChange, children, className = "" }: TabsProps) {
  const [internalValue, setInternalValue] = React.useState("")
  const value = controlledValue || internalValue
  const handleChange = onValueChange || setInternalValue

  return (
    <TabsContext.Provider value={{ value, onValueChange: handleChange }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  )
}

function TabsList({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`inline-flex h-9 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground ${className}`}>
      {children}
    </div>
  )
}

function TabsTrigger({ value, children, className = "" }: { value: string; children: React.ReactNode; className?: string }) {
  const ctx = React.useContext(TabsContext)
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${
        ctx.value === value ? "bg-background text-foreground shadow" : ""
      } ${className}`}
      onClick={() => ctx.onValueChange(value)}
    >
      {children}
    </button>
  )
}

function TabsContent({ value, children, className = "" }: { value: string; children: React.ReactNode; className?: string }) {
  const ctx = React.useContext(TabsContext)
  if (ctx.value !== value) return null
  return (
    <div className={`mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${className}`}>
      {children}
    </div>
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
