"use client"

import * as React from "react"

interface SheetContextType {
  open: boolean
  setOpen: React.Dispatch<React.SetStateAction<boolean>>
  titleId: string
}

const SheetContext = React.createContext<SheetContextType>({
  open: false,
  setOpen: () => {},
  titleId: "",
})

function Sheet({ children, open: controlledOpen, onOpenChange }: {
  children: React.ReactNode
  open?: boolean
  onOpenChange?: (open: boolean) => void
}) {
  const titleId = React.useId()
  const [internalOpen, setInternalOpen] = React.useState(false)
  const open = controlledOpen !== undefined ? controlledOpen : internalOpen
  const setOpen = React.useCallback((val: React.SetStateAction<boolean>) => {
    const next = typeof val === "function" ? val(open) : val
    setInternalOpen(next)
    onOpenChange?.(next)
  }, [open, onOpenChange])

  return (
    <SheetContext.Provider value={{ open, setOpen, titleId }}>
      {children}
    </SheetContext.Provider>
  )
}

function SheetTrigger({ children, asChild, className = "", ...props }: {
  children: React.ReactNode
  asChild?: boolean
  className?: string
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { setOpen } = React.useContext(SheetContext)
  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children as React.ReactElement<Record<string, unknown>>, {
      onClick: () => setOpen(true),
    })
  }
  return (
    <button onClick={() => setOpen(true)} className={className} {...props}>
      {children}
    </button>
  )
}

function SheetContent({ children, className = "", side = "right" }: {
  children: React.ReactNode
  className?: string
  side?: "top" | "right" | "bottom" | "left"
}) {
  const { open, setOpen, titleId } = React.useContext(SheetContext)
  const panelRef = React.useRef<HTMLDivElement>(null)

  const sideClasses: Record<string, string> = {
    top: "inset-x-0 top-0 h-auto max-h-[85vh] border-b rounded-b-lg",
    right: "inset-y-0 right-0 w-3/4 max-w-sm border-l",
    bottom: "inset-x-0 bottom-0 h-auto max-h-[85vh] border-t rounded-t-lg",
    left: "inset-y-0 left-0 w-3/4 max-w-sm border-r",
  }

  React.useEffect(() => {
    if (open) {
      const originalOverflow = document.body.style.overflow
      document.body.style.overflow = "hidden"
      panelRef.current?.focus()
      return () => {
        document.body.style.overflow = originalOverflow
      }
    }
  }, [open])

  React.useEffect(() => {
    if (!open) return
    function handleEscape(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false)
    }
    document.addEventListener("keydown", handleEscape)
    return () => document.removeEventListener("keydown", handleEscape)
  }, [open, setOpen])

  React.useEffect(() => {
    if (!open) return
    // Focus first focusable element inside the panel
    const panel = panelRef.current
    if (panel) {
      const focusable = panel.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
      if (focusable.length > 0) focusable[0].focus()
    }
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <div
        className="fixed inset-0 bg-black/80"
        onClick={() => setOpen(false)}
        aria-hidden="true"
      />
      <div
        ref={panelRef}
        tabIndex={-1}
        className={`fixed z-50 bg-background p-6 shadow-lg transition ease-in-out focus:outline-none ${sideClasses[side]} ${className}`}
      >
        {children}
      </div>
    </div>
  )
}

function SheetTitle({ children, className = "" }: {
  children: React.ReactNode
  className?: string
}) {
  const { titleId } = React.useContext(SheetContext)
  return <h2 id={titleId} className={`text-lg font-semibold ${className}`}>{children}</h2>
}

export { Sheet, SheetContent, SheetTrigger, SheetTitle }
