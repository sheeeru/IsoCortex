"use client"

import * as React from "react"

interface SelectProps {
  value?: string
  onValueChange?: (value: string) => void
  children: React.ReactNode
}

function Select({ value, onValueChange, children }: SelectProps) {
  return (
    <select
      value={value}
      onChange={(e) => onValueChange?.(e.target.value)}
      className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </select>
  )
}

function SelectTrigger({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return <div className={className}>{children}</div>
}

function SelectValue({ placeholder }: { placeholder?: string }) {
  return <span className="text-muted-foreground">{placeholder}</span>
}

function SelectContent({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}

function SelectItem({ value, children, className = "" }: { value: string; children: React.ReactNode; className?: string }) {
  return <option value={value} className={className}>{children}</option>
}

function SelectGroup({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}

function SelectLabel({ children }: { children: React.ReactNode }) {
  return <optgroup label={String(children)} />
}

export {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
}
