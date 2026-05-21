"use client"

import * as React from "react"

interface SelectProps {
  value?: string
  onValueChange?: (value: string) => void
  children: React.ReactNode
}

const SelectContext = React.createContext<{
  value: string
  onValueChange: (value: string) => void
  open: boolean
  setOpen: (open: boolean) => void
  listboxId: string
}>({ value: "", onValueChange: () => {}, open: false, setOpen: () => {}, listboxId: "" })

function Select({ value = "", onValueChange = () => {}, children }: SelectProps) {
  const [open, setOpen] = React.useState(false)
  const listboxId = React.useId()

  return (
    <SelectContext.Provider value={{ value, onValueChange, open, setOpen, listboxId }}>
      <div className="relative inline-block">{children}</div>
    </SelectContext.Provider>
  )
}

function SelectTrigger({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const { open, setOpen, listboxId } = React.useContext(SelectContext)
  return (
    <button
      type="button"
      role="combobox"
      aria-expanded={open}
      aria-controls={listboxId}
      aria-haspopup="listbox"
      onClick={() => setOpen(!open)}
      className={`flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring ${className}`}
    >
      {children}
    </button>
  )
}

function SelectValue({ placeholder }: { placeholder?: string }) {
  const { value } = React.useContext(SelectContext)
  return <span className={value ? "" : "text-muted-foreground"}>{value || placeholder}</span>
}

function SelectContent({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const { open, setOpen, listboxId } = React.useContext(SelectContext)
  const ref = React.useRef<HTMLDivElement>(null)
  const [focusedIndex, setFocusedIndex] = React.useState(-1)

  React.useEffect(() => {
    if (!open) return
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [open, setOpen])

  if (!open) return null

  function handleKeyDown(e: React.KeyboardEvent) {
    const items = Array.from(ref.current?.querySelectorAll<HTMLElement>('[role="option"]') || [])
    const itemCount = items.length

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault()
        setFocusedIndex(focusedIndex < itemCount - 1 ? focusedIndex + 1 : 0)
        items[focusedIndex < itemCount - 1 ? focusedIndex + 1 : 0]?.focus()
        break
      case "ArrowUp":
        e.preventDefault()
        setFocusedIndex(focusedIndex > 0 ? focusedIndex - 1 : itemCount - 1)
        items[focusedIndex > 0 ? focusedIndex - 1 : itemCount - 1]?.focus()
        break
      case "Enter":
      case " ":
        e.preventDefault()
        if (focusedIndex >= 0 && items[focusedIndex]) {
          items[focusedIndex].click()
        }
        break
      case "Escape":
        e.preventDefault()
        setOpen(false)
        break
      case "Tab":
        setOpen(false)
        break
    }
  }

  return (
    <div
      ref={ref}
      id={listboxId}
      role="listbox"
      className={`absolute top-full left-0 z-50 mt-1 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md ${className}`}
      onKeyDown={handleKeyDown}
    >
      <div className="p-1">{children}</div>
    </div>
  )
}

function SelectItem({ value, children, className = "" }: { value: string; children: React.ReactNode; className?: string }) {
  const ctx = React.useContext(SelectContext)
  const isSelected = ctx.value === value

  return (
    <div
      role="option"
      aria-selected={isSelected}
      tabIndex={0}
      className={`relative flex w-full cursor-pointer select-none items-center rounded-sm py-1.5 pl-2 pr-8 text-sm outline-none hover:bg-accent hover:text-accent-foreground ${
        isSelected ? "bg-accent text-accent-foreground" : ""
      } ${className}`}
      onClick={() => {
        ctx.onValueChange(value)
        ctx.setOpen(false)
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          ctx.onValueChange(value)
          ctx.setOpen(false)
        }
      }}
    >
      {children}
    </div>
  )
}

function SelectGroup({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}

function SelectLabel({ children }: { children: React.ReactNode }) {
  return <div className="px-2 py-1.5 text-sm font-semibold">{children}</div>
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
