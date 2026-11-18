/**
 * ZONOS3-Mini — desktop half.
 *
 * One status-bar chip reporting whether the ZONOS3 engine is resident, plus a
 * menu that sets how long it stays loaded after the last utterance.
 *
 * Data: GET /api/plugins/zonos3-tts/status — the JSON file the agent half writes
 * on every lease/residency change. The chip renders in the dashboard process;
 * the engine and its leases live in the agent process, so that file is the
 * handoff. POST /api/plugins/zonos3-tts/settings persists a new window, and the
 * agent re-reads it live (`_live_setting_fn`), so no restart is needed.
 *
 * Plain ESM, loaded uncompiled — UI is jsx() calls, not JSX syntax.
 *
 * SDK contract, verified against apps/desktop/src/sdk/index.ts:
 *   - `jsx` is NOT an SDK export; it comes from 'react/jsx-runtime'.
 *   - Only five DropdownMenu parts ship (Menu/Content/Item/Separator/Trigger).
 *     There is no `DropdownMenuLabel` — a styled div is the section label.
 *   - `useAtomValue` does not exist; the atom subscription is `useValue`.
 *     `useQuery` is the sanctioned polling layer.
 *   - `ctx.rest(path, opts)` is the only door to this plugin's backend
 *     namespace. `PluginRestOptions` is `{ method, body, upload, timeoutMs }`.
 */

import {
  cn,
  Codicon,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  useQuery
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx } from 'react/jsx-runtime'

const ID = 'zonos3-tts'
const POLL_MS = 4000
const MENU_LABEL =
  'px-2 py-1 text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground'
const NOTE = 'px-2 pb-1 text-xs text-muted-foreground'

const WINDOWS = [
  { minutes: 5, seconds: 300 },
  { minutes: 10, seconds: 600 },
  { minutes: 15, seconds: 900 }
]

/** Render the engine state as a short chip label. */
function label(state) {
  if (!state || !state.ok) {
    // A stale or dead agent half used to be indistinguishable from a healthy
    // idle engine. Name the difference instead of implying "unloaded".
    return state && state.reason ? 'ZONOS3 offline' : 'ZONOS3'
  }
  const tts = state.tts
  if (tts === 'loading' || tts === 'starting') return 'ZONOS3 warming…'
  if (tts === 'unloading') return 'ZONOS3 stopping…'
  if (tts === 'warm') return 'ZONOS3 warm'
  if (tts === 'unloaded') return 'ZONOS3 unloaded'
  return `ZONOS3 ${tts || 'unknown'}`
}

/** "5 min" / "90 s" — a window, phrased for a menu row. */
function windowText(seconds) {
  const s = Number(seconds)
  if (!Number.isFinite(s) || s <= 0) return 'never'
  if (s % 60 === 0) return `${s / 60} min`
  return `${Math.round(s)} s`
}

function ZonosChip({ ctx }) {
  const status = useQuery({
    queryKey: [ID, 'status'],
    queryFn: () => ctx.rest('/status'),
    refetchInterval: POLL_MS
  })
  const [busyUnload, setBusy] = useState(false)
  const [unloadError, setError] = useState('')
  const [busyStt, setBusyStt] = useState(false)
  const [sttError, setSttError] = useState('')

  // Clear a stale refusal as soon as the engine state changes.
  useEffect(() => {
    setError('')
  }, [status.data && status.data.tts])

  const state = status.data || { ok: false }
  const offline = !state.ok && Boolean(state.reason)
  const warm = state.ok && state.tts === 'warm'
  const busy = state.ok && (state.tts === 'loading' || state.tts === 'starting'
    || state.tts === 'unloading')
  const current = state.ok ? state.unload_after : undefined

  const save = (patch) => {
    ctx
      .rest('/settings', { method: 'POST', body: patch })
      .then(() => status.refetch())
      .catch(() => {})
  }

  const unload = () => {
    setBusy(true)
    ctx
      .rest('/unload', { method: 'POST' })
      .then(() => {
        // The engine stops on the agent's next poll (~1 s), so poll a little
        // faster for a moment rather than refetching once and showing "warm".
        let tries = 0
        const tick = () => {
          status.refetch().finally(() => {
            tries += 1
            if (tries < 8) window.setTimeout(tick, 700)
            else setBusy(false)
          })
        }
        window.setTimeout(tick, 400)
      })
      .catch((err) => {
        setBusy(false)
        setError(String((err && err.message) || err || 'unload failed'))
      })
  }

  const unloadStt = () => {
    setBusyStt(true)
    ctx
      .rest('/stt/unload', { method: 'POST' })
      .then(() => {
        let tries = 0
        const tick = () => {
          status.refetch().finally(() => {
            tries += 1
            if (tries < 8) window.setTimeout(tick, 700)
            else setBusyStt(false)
          })
        }
        window.setTimeout(tick, 400)
      })
      .catch((err) => {
        setBusyStt(false)
        setSttError(String((err && err.message) || err || 'unload failed'))
      })
  }

  return jsx(DropdownMenu, {
    children: [
      jsx(DropdownMenuTrigger, {
        asChild: true,
        children: jsx('button', {
          type: 'button',
          className: cn(
            'hermes-zonos-chip inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem]',
            'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
            warm && 'hermes-zonos-chip-hot',
            busy && 'hermes-zonos-chip-busy',
            offline && 'hermes-zonos-chip-offline'
          ),
          title: offline ? `ZONOS3 agent half: ${state.reason}` : undefined,
          children: [
            jsx(Codicon, {
              name: busy ? 'sync' : warm ? 'unmute' : 'mute',
              spinning: Boolean(busy),
              className: 'hermes-zonos-chip-icon'
            }),
            jsx('span', { className: 'hermes-zonos-chip-label', children: label(state) })
          ]
        })
      }),
      jsx(DropdownMenuContent, {
        align: 'end',
        className: 'w-64',
        children: [
          jsx('div', { className: MENU_LABEL, children: 'ZONOS3-Mini engine' }),
          offline
            ? jsx('div', { className: NOTE, children: `No agent reporting: ${state.reason}` })
            : jsx('div', {
              className: `${NOTE} space-y-1`,
              children: [
                jsx('div', {
                  children: `State: ${state.tts}${state.tts_leased ? ' (leased)' : ''}`
                }),
                state.ok && state.pid
                  ? jsx('div', { children: `Process: ${state.pid}` })
                  : null,
                state.ok && state.voices
                  ? jsx('div', { children: `Voice: ${state.voice || '—'} (${state.voices} available)` })
                  : null,
                state.ok
                  ? jsx('div', {
                    children: state.streams_pcm
                      ? `Streaming: native ${state.stream_sample_rate} Hz`
                      : 'Streaming: per-sentence (buffered)'
                  })
                  : null,
                state.ok && state.stt
                  ? jsx('div', {
                    children: `STT: ${state.stt.loaded ? 'loaded' : 'unloaded'}`
                      + ` (${state.stt.engine || 'onnx'}, CPU)`
                  })
                  : null,
                state.ok && state.last_error
                  ? jsx('div', { children: `Last error: ${state.last_error}` })
                  : null
              ]
            }),
          jsx(DropdownMenuSeparator, {}),
          jsx('div', { className: MENU_LABEL, children: 'Unload after idle' }),
          jsx('div', {
            className: NOTE,
            children: state.ok
              ? `Now: ${windowText(current)} after the last utterance`
              : 'Waiting for the agent half'
          }),
          ...WINDOWS.map((w) =>
            jsx(DropdownMenuItem, {
              key: w.seconds,
              onSelect: () => save({ unload_after_seconds: w.seconds }),
              children: [
                jsx(Codicon, {
                  name: Number(current) === w.seconds ? 'check' : 'blank',
                  className: 'mr-2 text-xs'
                }),
                `Keep loaded ${w.minutes} min`
              ]
            })
          ),
          jsx(DropdownMenuSeparator, {}),
          jsx(DropdownMenuItem, {
            onSelect: () => save({ unload_after_seconds: 60 }),
            children: [
              jsx(Codicon, {
                name: Number(current) === 60 ? 'check' : 'blank',
                className: 'mr-2 text-xs'
              }),
              'Keep loaded 1 min'
            ]
          }),
          jsx('div', {
            className: NOTE,
            children: 'Applies to the next idle release (the agent re-reads this window within ~2 s).'
          }),
          jsx(DropdownMenuSeparator, {}),
          jsx('div', { className: MENU_LABEL, children: 'Free VRAM now' }),
          jsx(DropdownMenuItem, {
            disabled: busyUnload || !warm || Boolean(state.ok && state.tts_leased),
            onSelect: () => unload(),
            children: [
              jsx(Codicon, {
                name: busyUnload ? 'sync' : 'debug-stop',
                spinning: Boolean(busyUnload),
                className: 'mr-2 text-xs'
              }),
              busyUnload
                ? 'Unloading…'
                : warm
                  ? 'Unload model now'
                  : 'Not loaded'
            ]
          }),
          unloadError
            ? jsx('div', { className: NOTE, children: unloadError })
            : state.ok && state.tts_leased && warm
              ? jsx('div', {
                className: NOTE,
                children: 'A surface holds the engine — release it (stop read-aloud) first.'
              })
              : jsx('div', {
                className: NOTE,
                children: 'Stops the engine process: reclaims weights, KV cache and CUDA context.'
              }),
          jsx(DropdownMenuSeparator, {}),
          jsx('div', { className: MENU_LABEL, children: 'Speech recognition' }),
          jsx(DropdownMenuItem, {
            disabled: busyStt || !state.ok || !state.stt
              || state.stt.loaded === false || Boolean(state.stt_leased),
            onSelect: () => unloadStt(),
            children: [
              jsx(Codicon, {
                name: busyStt ? 'sync' : 'debug-stop',
                spinning: Boolean(busyStt),
                className: 'mr-2 text-xs'
              }),
              busyStt
                ? 'Releasing…'
                : state.stt
                  ? (state.stt.loaded === false ? 'STT not loaded' : 'Unload STT (free RAM)')
                  : 'STT not registered'
            ]
          }),
          sttError
            ? jsx('div', { className: NOTE, children: sttError })
            : state.stt_leased
              ? jsx('div', { className: NOTE, children: 'A recording surface holds the recogniser.' })
              : jsx('div', {
                className: NOTE,
                children: state.stt
                  ? 'SenseVoice is CPU-only: this frees ~323 MB RAM, no VRAM. The voice engine stays loaded.'
                  : 'Enable stt_enabled and set stt.provider: sensevoice to use it.'
              })
        ]
      })
    ]
  })
}

const CHIP_CSS = `
.hermes-zonos-chip { white-space: nowrap; }
.hermes-zonos-chip-icon { font-size: 12px; opacity: .7; }
.hermes-zonos-chip-hot .hermes-zonos-chip-icon { opacity: 1; }
.hermes-zonos-chip-busy .hermes-zonos-chip-icon { opacity: 1; }
.hermes-zonos-chip-offline .hermes-zonos-chip-label { opacity: .6; }
.hermes-zonos-chip-label { white-space: nowrap; }
`

export default {
  id: ID,
  name: 'ZONOS3-Mini',
  defaultEnabled: true,
  register(ctx) {
    const style = document.createElement('style')
    style.textContent = CHIP_CSS
    document.head.append(style)
    ctx.onDispose(() => style.remove())

    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 31,
      render: () => jsx(ZonosChip, { ctx })
    })
  }
}
