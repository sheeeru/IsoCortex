"use client"

import * as React from "react"

const TOAST_LIMIT = 5
const TOAST_REMOVE_DELAY = 5000

type ToastType = {
  id: string
  title?: string
  description?: string
  variant?: "default" | "destructive"
}

type Action =
  | { type: "ADD_TOAST"; toast: ToastType }
  | { type: "UPDATE_TOAST"; toast: Partial<ToastType> }
  | { type: "DISMISS_TOAST"; toastId?: string }
  | { type: "REMOVE_TOAST"; toastId?: string }

let count = 0

function genId() {
  count = (count + 1) % Number.MAX_SAFE_INTEGER
  return count.toString()
}

const toastTimeouts = new Map<string, ReturnType<typeof setTimeout>>()

function addToRemoveQueue(toastId: string) {
  if (toastTimeouts.has(toastId)) return
  const timeout = setTimeout(() => {
    toastTimeouts.delete(toastId)
    dispatch({ type: "REMOVE_TOAST", toastId })
  }, TOAST_REMOVE_DELAY)
  toastTimeouts.set(toastId, timeout)
}

const reducer = (state: ToastType[], action: Action): ToastType[] => {
  switch (action.type) {
    case "ADD_TOAST":
      return [action.toast, ...state].slice(0, TOAST_LIMIT)
    case "UPDATE_TOAST":
      return state.map((t) => (t.id === action.toast.id ? { ...t, ...action.toast } : t))
    case "DISMISS_TOAST": {
      const id = action.toastId
      if (id) addToRemoveQueue(id)
      else state.forEach((t) => addToRemoveQueue(t.id))
      return state
    }
    case "REMOVE_TOAST":
      return action.toastId ? state.filter((t) => t.id !== action.toastId) : []
    default:
      return state
  }
}

const listeners: Array<(state: ToastType[]) => void> = []
let memoryState: ToastType[] = []

function dispatch(action: Action) {
  memoryState = reducer(memoryState, action)
  listeners.forEach((listener) => listener(memoryState))
}

function toast({ ...props }: Omit<ToastType, "id">) {
  const id = genId()
  dispatch({ type: "ADD_TOAST", toast: { ...props, id } })
  addToRemoveQueue(id)
  return { id, dismiss: () => dispatch({ type: "DISMISS_TOAST", toastId: id }) }
}

function useToast() {
  const [state, setState] = React.useState<ToastType[]>(memoryState)

  React.useEffect(() => {
    listeners.push(setState)
    return () => {
      const index = listeners.indexOf(setState)
      if (index > -1) listeners.splice(index, 1)
    }
  }, [])

  return {
    toasts: state,
    toast,
    dismiss: (toastId?: string) => dispatch({ type: "DISMISS_TOAST", toastId }),
  }
}

export { useToast, toast }
export type { ToastType }
