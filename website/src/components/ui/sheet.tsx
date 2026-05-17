"use client"

import * as React from "react"

interface SheetContextType {
  open: boolean
  setOpen: React.Dispatch<React.SetStateAction<boolean>>
}

const SheetContext = React.createContext<SheetContextType>({
  open: false,
  setOpen: () => {},
})

function Sheet({ children, open: controlledOpen, onOpenChange }: {
  children: React.ReactNode
  open?: boolean
  onOpenChange?: (open: boolean) => void
}) {
  const [internalOpen, setInternalOpen] = React.useState(false)
  const open = controlledOpen !== undefined ? controlledOpen : internalOpen
  const setOpen = React.useCallback((val: React.SetStateAction<boolean>) => {
    const next = typeof val === "function" ? val(open) : val
    setInternalOpen(next)
    onOpenChange?.(next)
  }, [open, onOpenChange])

  return (
    <SheetContext.Provider value={{ open, setOpen }}>
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
    return React.cloneElement(children as React.ReactElement<any>, {
      onClick: () => setOpen(true),
    })
  }
  return (
    <button onClick={() => setOpen(true)} className={className} {...props}>
      {children}
    </button>
  )
}

function SheetContent({ children, className = "" }: {
  children: React.ReactNode
  className?: string
}) {
  const { open, setOpen } = React.useContext(SheetContext)
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50">
      <div className="fixed inset-0 bg-black/80" onClick={() => setOpen(false)} />
      <div className={`fixed inset-y-0 right-0 z-50 w-3/4 max-w-sm border-l bg-background p-6 shadow-lg transition ease-in-out ${className}`}>
        {children}
      </div>
    </div>
  )
}

function SheetTitle({ children, className = "" }: {
  children: React.ReactNode
  className?: string
}) {
  return <h2 className={`text-lg font-semibold ${className}`}>{children}</h2>
}

export { Sheet, SheetContent, SheetTrigger, SheetTitle }
