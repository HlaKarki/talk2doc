import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useMemo, useRef, useState } from 'react'

type DocumentItem = {
  id: string
  filename: string
  file_type: string
  status: string
}

type DatasetItem = {
  id: string
  filename: string
  file_type: string
  status: string
  row_count?: number | null
  column_count?: number | null
}

type DatasetListResponse = {
  datasets: DatasetItem[]
  total: number
}

type ChatApiResponse = {
  response: string
  intent: string
  agent_used: string
  conversation_id?: string
  metadata?: Record<string, unknown>
}

type PlotlyFigure = {
  data: unknown[]
  layout?: Record<string, unknown>
}

type ChatVisualization = {
  type?: string
  column?: string
  x_column?: string
  y_column?: string
  date_column?: string
  value_column?: string
  figure: PlotlyFigure
}

type MemoryStatsResponse = {
  long_term?: {
    total_count?: number
    by_type?: Record<string, number>
  }
  semantic?: {
    total_count?: number
    by_intent?: Record<string, number>
    by_agent?: Record<string, number>
  }
  graph?: {
    total_nodes?: number
    total_edges?: number
  }
}

type KnowledgeGraphStatus = {
  status: 'idle' | 'available' | 'missing' | 'error'
  nodeCount?: number
  edgeCount?: number
  message?: string
}

type ModelListItem = {
  id: string
  algorithm: string
  target_column: string
  created_at: string
}

type ModelsListResponse = {
  models: ModelListItem[]
  total: number
}

type ConversationListItem = {
  id: string
  title?: string | null
  updated_at: string
  message_count: number
}

type ChatMessage = {
  role: 'user' | 'assistant'
  text: string
  intent?: string
  agent?: string
  visualizations?: ChatVisualization[]
}

type ContextSelection = {
  type: 'document' | 'dataset'
  id: string
  label: string
}

const API_BASE_URL =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  (import.meta.env.SERVER_URL as string | undefined) ??
  'http://localhost:8000'

const PLOTLY_SCRIPT_ID = 'plotly-cdn-script'
const PLOTLY_SCRIPT_SRC = 'https://cdn.plot.ly/plotly-2.35.2.min.js'

let plotlyLoadPromise: Promise<void> | null = null

declare global {
  interface Window {
    Plotly?: {
      newPlot: (
        root: HTMLElement,
        data: unknown[],
        layout?: Record<string, unknown>,
        config?: Record<string, unknown>,
      ) => Promise<void> | void
      purge: (root: HTMLElement) => void
    }
  }
}

export const Route = createFileRoute('/')({ component: ChatPage })

function ChatPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [datasets, setDatasets] = useState<DatasetItem[]>([])
  const [selection, setSelection] = useState<ContextSelection | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [isLoadingLists, setIsLoadingLists] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState<'document' | 'dataset' | null>(null)
  const [loadingDots, setLoadingDots] = useState('.')
  const [showcaseLoading, setShowcaseLoading] = useState(false)
  const [showcaseError, setShowcaseError] = useState<string | null>(null)
  const [memoryStats, setMemoryStats] = useState<MemoryStatsResponse | null>(null)
  const [modelSummary, setModelSummary] = useState<ModelsListResponse | null>(null)
  const [recentConversations, setRecentConversations] = useState<
    ConversationListItem[]
  >([])
  const [knowledgeGraphStatus, setKnowledgeGraphStatus] =
    useState<KnowledgeGraphStatus>({
      status: 'idle',
      message: 'Select a document to inspect knowledge graph status.',
    })
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    void refreshSources()
    void refreshShowcaseOverview()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!isSending) {
      setLoadingDots('.')
      return
    }

    const id = window.setInterval(() => {
      setLoadingDots((prev) => (prev.length >= 3 ? '.' : `${prev}.`))
    }, 350)

    return () => window.clearInterval(id)
  }, [isSending])

  useEffect(() => {
    void refreshKnowledgeGraphStatus()
  }, [selection])

  async function refreshSources() {
    setIsLoadingLists(true)
    setError(null)

    try {
      const [docRes, datasetRes] = await Promise.all([
        fetch(`${API_BASE_URL}/documents`),
        fetch(`${API_BASE_URL}/datasets`),
      ])

      if (!docRes.ok) {
        throw new Error(`Failed to load documents (${docRes.status})`)
      }
      if (!datasetRes.ok) {
        throw new Error(`Failed to load datasets (${datasetRes.status})`)
      }

      const docs = (await docRes.json()) as DocumentItem[]
      const datasetPayload = (await datasetRes.json()) as DatasetListResponse

      setDocuments(docs)
      setDatasets(datasetPayload.datasets ?? [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load data sources')
    } finally {
      setIsLoadingLists(false)
    }
  }

  async function refreshShowcaseOverview() {
    setShowcaseLoading(true)
    setShowcaseError(null)

    try {
      const [memoryRes, modelsRes, conversationsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/memory/stats`),
        fetch(`${API_BASE_URL}/models`),
        fetch(`${API_BASE_URL}/conversations?limit=5`),
      ])

      if (!memoryRes.ok) {
        throw new Error(`Failed to load memory stats (${memoryRes.status})`)
      }
      if (!modelsRes.ok) {
        throw new Error(`Failed to load model summary (${modelsRes.status})`)
      }
      if (!conversationsRes.ok) {
        throw new Error(
          `Failed to load conversation summary (${conversationsRes.status})`,
        )
      }

      setMemoryStats((await memoryRes.json()) as MemoryStatsResponse)
      setModelSummary((await modelsRes.json()) as ModelsListResponse)
      setRecentConversations(
        (await conversationsRes.json()) as ConversationListItem[],
      )
    } catch (e) {
      setShowcaseError(
        e instanceof Error ? e.message : 'Failed to load showcase data',
      )
    } finally {
      setShowcaseLoading(false)
    }
  }

  async function refreshKnowledgeGraphStatus() {
    if (!selection || selection.type !== 'document') {
      setKnowledgeGraphStatus({
        status: 'idle',
        message: 'Select a document to inspect knowledge graph status.',
      })
      return
    }

    try {
      const res = await fetch(`${API_BASE_URL}/kg/documents/${selection.id}/graph`)

      if (res.status === 404) {
        setKnowledgeGraphStatus({
          status: 'missing',
          message: 'No graph extracted yet for this document.',
        })
        return
      }

      if (!res.ok) {
        throw new Error(`KG status request failed (${res.status})`)
      }

      const payload = (await res.json()) as {
        node_count?: number
        edge_count?: number
      }

      setKnowledgeGraphStatus({
        status: 'available',
        nodeCount: payload.node_count ?? 0,
        edgeCount: payload.edge_count ?? 0,
      })
    } catch (e) {
      setKnowledgeGraphStatus({
        status: 'error',
        message: e instanceof Error ? e.message : 'Failed to load KG status',
      })
    }
  }

  async function onUploadDocument(file: File) {
    setUploading('document')
    setError(null)

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const detail = await readErrorDetail(res)
        throw new Error(`Document upload failed: ${detail}`)
      }

      await refreshSources()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Document upload failed')
    } finally {
      setUploading(null)
    }
  }

  async function onUploadDataset(file: File) {
    setUploading('dataset')
    setError(null)

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch(`${API_BASE_URL}/datasets/upload`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const detail = await readErrorDetail(res)
        throw new Error(`Dataset upload failed: ${detail}`)
      }

      await refreshSources()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Dataset upload failed')
    } finally {
      setUploading(null)
    }
  }

  async function sendMessage() {
    const query = input.trim()
    if (!query || isSending) return

    setIsSending(true)
    setError(null)
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', text: query }])

    const payload: Record<string, string> = { query }
    if (conversationId) payload.conversation_id = conversationId
    if (selection?.type === 'document') payload.document_id = selection.id
    if (selection?.type === 'dataset') payload.dataset_id = selection.id

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const detail = await readErrorDetail(res)
        throw new Error(`Chat request failed: ${detail}`)
      }

      const data = (await res.json()) as ChatApiResponse
      if (data.conversation_id) {
        setConversationId(data.conversation_id)
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: data.response,
          intent: data.intent,
          agent: data.agent_used,
          visualizations: extractVisualizations(data.metadata),
        },
      ])
    } catch (e) {
      const msg =
        e instanceof Error ? e.message : 'Failed to send chat message'
      setError(msg)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Error: ${msg}` },
      ])
    } finally {
      setIsSending(false)
      void refreshShowcaseOverview()
      void refreshKnowledgeGraphStatus()
    }
  }

  function startNewChat() {
    setConversationId(null)
    setMessages([])
  }

  function clearContextAndStartNewChat() {
    setSelection(null)
    setConversationId(null)
    setMessages([])
  }

  const selectionLabel = useMemo(() => {
    if (!selection) return 'None'
    return `${selection.type}: ${selection.label}`
  }, [selection])

  return (
    <main className="h-[calc(100vh-57px)] bg-white text-black">
      <div className="grid h-full grid-cols-1 md:grid-cols-[320px_1fr]">
        <aside className="border-b border-black p-3 md:border-b-0 md:border-r md:overflow-y-auto">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase">Sources</h2>
            <button
              type="button"
              onClick={() => void refreshSources()}
              className="border border-black px-2 py-1 text-xs disabled:opacity-50"
              disabled={isLoadingLists}
            >
              {isLoadingLists ? 'Loading...' : 'Refresh'}
            </button>
          </div>

          <section className="mb-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Documents</h3>
              <label className="cursor-pointer border border-black px-2 py-1 text-xs">
                Upload
                <input
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) void onUploadDocument(file)
                    e.currentTarget.value = ''
                  }}
                  disabled={uploading !== null}
                />
              </label>
            </div>
            <ul className="space-y-2">
              {documents.map((doc) => (
                <li
                  key={doc.id}
                  className={`border p-2 text-sm ${
                    selection?.type === 'document' && selection.id === doc.id
                      ? 'border-black bg-black text-white'
                      : 'border-black bg-white text-black'
                  }`}
                >
                  <div className="truncate font-medium">{doc.filename}</div>
                  <div className="text-xs">
                    {doc.file_type} | {doc.status}
                  </div>
                  {selection?.type === 'document' && selection.id === doc.id && (
                    <div className="mt-1 text-xs font-semibold">Selected</div>
                  )}
                  <button
                    type="button"
                    className={`mt-2 border px-2 py-1 text-xs ${
                      selection?.type === 'document' && selection.id === doc.id
                        ? 'border-white bg-white text-black'
                        : 'border-black bg-white text-black'
                    }`}
                    onClick={() =>
                      setSelection({
                        type: 'document',
                        id: doc.id,
                        label: doc.filename,
                      })
                    }
                    disabled={
                      selection?.type === 'document' && selection.id === doc.id
                    }
                  >
                    {selection?.type === 'document' && selection.id === doc.id
                      ? 'In Chat'
                      : 'Use in Chat'}
                  </button>
                </li>
              ))}
              {documents.length === 0 && (
                <li className="border border-black p-2 text-xs">
                  No documents found.
                </li>
              )}
            </ul>
          </section>

          <section className="mb-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Datasets</h3>
              <label className="cursor-pointer border border-black px-2 py-1 text-xs">
                Upload
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) void onUploadDataset(file)
                    e.currentTarget.value = ''
                  }}
                  disabled={uploading !== null}
                />
              </label>
            </div>
            <ul className="space-y-2">
              {datasets.map((dataset) => (
                <li
                  key={dataset.id}
                  className={`border p-2 text-sm ${
                    selection?.type === 'dataset' && selection.id === dataset.id
                      ? 'border-black bg-black text-white'
                      : 'border-black bg-white text-black'
                  }`}
                >
                  <div className="truncate font-medium">{dataset.filename}</div>
                  <div className="text-xs">
                    {dataset.file_type} | {dataset.row_count ?? '?'} rows |{' '}
                    {dataset.column_count ?? '?'} cols
                  </div>
                  {selection?.type === 'dataset' &&
                    selection.id === dataset.id && (
                      <div className="mt-1 text-xs font-semibold">Selected</div>
                    )}
                  <button
                    type="button"
                    className={`mt-2 border px-2 py-1 text-xs ${
                      selection?.type === 'dataset' && selection.id === dataset.id
                        ? 'border-white bg-white text-black'
                        : 'border-black bg-white text-black'
                    }`}
                    onClick={() =>
                      setSelection({
                        type: 'dataset',
                        id: dataset.id,
                        label: dataset.filename,
                      })
                    }
                    disabled={
                      selection?.type === 'dataset' && selection.id === dataset.id
                    }
                  >
                    {selection?.type === 'dataset' && selection.id === dataset.id
                      ? 'In Chat'
                      : 'Use in Chat'}
                  </button>
                </li>
              ))}
              {datasets.length === 0 && (
                <li className="border border-black p-2 text-xs">
                  No datasets found.
                </li>
              )}
            </ul>
          </section>

          <section className="border border-black p-2 text-xs">
            <div>
              <span className="font-semibold">Current context:</span>{' '}
              {selectionLabel}
            </div>
            <button
              type="button"
              className="mt-2 border border-black px-2 py-1"
              onClick={clearContextAndStartNewChat}
            >
              Clear context (new chat)
            </button>
          </section>
        </aside>

        <section className="flex min-h-0 flex-col p-3">
          <div className="mb-2 flex items-center justify-between gap-2 border border-black p-2 text-xs">
            <div className="truncate">
              <span className="font-semibold">Conversation:</span>{' '}
              {conversationId ?? 'new'}
            </div>
            <button
              type="button"
              className="border border-black px-2 py-1"
              onClick={startNewChat}
            >
              New chat
            </button>
          </div>

          <div className="mb-2 border border-black p-2 text-xs">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-semibold uppercase">Backend Showcase</span>
              <button
                type="button"
                className="border border-black px-2 py-1 disabled:opacity-50"
                onClick={() => {
                  void refreshShowcaseOverview()
                  void refreshKnowledgeGraphStatus()
                }}
                disabled={showcaseLoading}
              >
                {showcaseLoading ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
              <div className="border border-black p-2">
                <div className="mb-1 font-semibold">Memory</div>
                <div>Long-term: {memoryStats?.long_term?.total_count ?? 0}</div>
                <div>Semantic: {memoryStats?.semantic?.total_count ?? 0}</div>
                <div>Memory graph nodes: {memoryStats?.graph?.total_nodes ?? 0}</div>
              </div>

              <div className="border border-black p-2">
                <div className="mb-1 font-semibold">Knowledge Graph</div>
                {knowledgeGraphStatus.status === 'available' ? (
                  <>
                    <div>Nodes: {knowledgeGraphStatus.nodeCount ?? 0}</div>
                    <div>Edges: {knowledgeGraphStatus.edgeCount ?? 0}</div>
                  </>
                ) : (
                  <div>{knowledgeGraphStatus.message ?? 'Unavailable'}</div>
                )}
              </div>

              <div className="border border-black p-2">
                <div className="mb-1 font-semibold">Models</div>
                <div>Trained models: {modelSummary?.total ?? 0}</div>
                {modelSummary?.models?.[0] && (
                  <div className="truncate">
                    Latest: {modelSummary.models[0].algorithm} (
                    {modelSummary.models[0].target_column})
                  </div>
                )}
              </div>

              <div className="border border-black p-2">
                <div className="mb-1 font-semibold">Conversations</div>
                <div>Recent sessions: {recentConversations.length}</div>
                {recentConversations.slice(0, 2).map((conv) => (
                  <div key={conv.id} className="truncate">
                    {conv.title || 'Untitled'} ({conv.message_count} msgs)
                  </div>
                ))}
              </div>
            </div>

            {showcaseError && (
              <div className="mt-2 border border-black p-1">
                Showcase error: {showcaseError}
              </div>
            )}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto border border-black p-3">
            {messages.length === 0 && (
              <p className="text-sm">
                Ask a question. Select a document or dataset on the left to
                scope context.
              </p>
            )}

            <div className="space-y-3">
              {messages.map((m, idx) => (
                <div key={`${m.role}-${idx}`} className="border border-black p-2">
                  <div className="mb-1 text-xs font-semibold uppercase">
                    {m.role}
                  </div>
                  <div className="whitespace-pre-wrap text-sm">{m.text}</div>
                  {m.role === 'assistant' &&
                    m.visualizations &&
                    m.visualizations.length > 0 && (
                      <div className="mt-2">
                        {m.visualizations.map((viz, vizIdx) => (
                          <PlotlyChart
                            key={`${m.role}-${idx}-viz-${vizIdx}`}
                            visualization={viz}
                            index={vizIdx}
                          />
                        ))}
                      </div>
                    )}
                  {m.role === 'assistant' && (m.intent || m.agent) && (
                    <div className="mt-2 text-xs">
                      {m.intent ? `intent: ${m.intent}` : ''}{' '}
                      {m.agent ? `agent: ${m.agent}` : ''}
                    </div>
                  )}
                </div>
              ))}

              {isSending && (
                <div className="border border-black p-2">
                  <div className="mb-1 text-xs font-semibold uppercase">
                    assistant
                  </div>
                  <div className="text-sm">{loadingDots}</div>
                </div>
              )}
            </div>
            <div ref={messagesEndRef} />
          </div>

          <div className="mt-2 flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void sendMessage()
                }
              }}
              placeholder="Type your message..."
              className="flex-1 border border-black px-3 py-2 text-sm outline-none"
              disabled={isSending}
            />
            <button
              type="button"
              onClick={() => void sendMessage()}
              className="border border-black px-4 py-2 text-sm disabled:opacity-50"
              disabled={isSending || input.trim().length === 0}
            >
              {isSending ? 'Sending...' : 'Send'}
            </button>
          </div>

          {error && (
            <p className="mt-2 border border-black p-2 text-xs">Error: {error}</p>
          )}
        </section>
      </div>
    </main>
  )
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const payload = (await res.json()) as { detail?: string }
    return payload.detail ?? `${res.status}`
  } catch {
    return `${res.status}`
  }
}

function extractVisualizations(
  metadata: Record<string, unknown> | undefined,
): ChatVisualization[] {
  if (!metadata) return []

  const rawVisualizations = metadata.visualizations
  if (!Array.isArray(rawVisualizations)) return []

  return rawVisualizations.filter(isVisualization)
}

function isVisualization(value: unknown): value is ChatVisualization {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  if (!item.figure || typeof item.figure !== 'object') return false
  const figure = item.figure as Record<string, unknown>
  return Array.isArray(figure.data)
}

function getVisualizationTitle(
  visualization: ChatVisualization,
  index: number,
): string {
  const label = visualization.type
    ? visualization.type.replace(/_/g, ' ')
    : `chart ${index + 1}`

  if (visualization.column) return `${label}: ${visualization.column}`
  if (visualization.x_column && visualization.y_column) {
    return `${label}: ${visualization.y_column} vs ${visualization.x_column}`
  }
  if (visualization.date_column && visualization.value_column) {
    return `${label}: ${visualization.value_column} over ${visualization.date_column}`
  }
  return label
}

function loadPlotlyScript(): Promise<void> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Plot rendering is only available in browser'))
  }

  if (window.Plotly) {
    return Promise.resolve()
  }

  if (plotlyLoadPromise) {
    return plotlyLoadPromise
  }

  plotlyLoadPromise = new Promise<void>((resolve, reject) => {
    const existing = document.getElementById(
      PLOTLY_SCRIPT_ID,
    ) as HTMLScriptElement | null

    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true })
      existing.addEventListener(
        'error',
        () => reject(new Error('Failed to load chart renderer')),
        { once: true },
      )
      return
    }

    const script = document.createElement('script')
    script.id = PLOTLY_SCRIPT_ID
    script.src = PLOTLY_SCRIPT_SRC
    script.async = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Failed to load chart renderer'))
    document.head.appendChild(script)
  })

  return plotlyLoadPromise
}

function PlotlyChart({
  visualization,
  index,
}: {
  visualization: ChatVisualization
  index: number
}) {
  const rootRef = useRef<HTMLDivElement | null>(null)
  const [renderError, setRenderError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const figure = visualization.figure

    setRenderError(null)

    void loadPlotlyScript()
      .then(() => {
        if (cancelled || !rootRef.current || !window.Plotly) return

        const layout = {
          ...(figure.layout ?? {}),
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
          font: { color: '#000000' },
        }

        return window.Plotly.newPlot(rootRef.current, figure.data, layout, {
          responsive: true,
          displayModeBar: false,
        })
      })
      .catch((e) => {
        if (!cancelled) {
          setRenderError(
            e instanceof Error ? e.message : 'Failed to render chart',
          )
        }
      })

    return () => {
      cancelled = true
      if (rootRef.current && window.Plotly) {
        window.Plotly.purge(rootRef.current)
      }
    }
  }, [visualization])

  return (
    <div className="mb-2 border border-black p-2">
      <div className="mb-2 text-xs font-semibold uppercase">
        {getVisualizationTitle(visualization, index)}
      </div>
      {renderError ? (
        <div className="text-xs">Chart error: {renderError}</div>
      ) : (
        <div ref={rootRef} className="h-[320px] w-full" />
      )}
    </div>
  )
}
