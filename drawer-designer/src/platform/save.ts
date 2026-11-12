/**
 * Handing a generated file to the user.
 *
 * In an ordinary browser tab this is the usual click on a hidden anchor. A
 * copy of the app published as a Claude artifact runs in a sandboxed frame
 * where that is blocked outright, and has to pass the file to the host
 * instead. That host accepts only a fixed list of extensions, and `.stl` is
 * not on it, so a lone STL is wrapped in a ZIP there rather than failing.
 *
 * Everything here degrades: with no host present the anchor is used, and the
 * app behaves exactly as it does when you run it yourself.
 */
import { tr, useUi } from '../store/uiStore';

/** Extensions the artifact host will save. */
const HOST_EXTENSIONS = new Set([
  'gif', 'png', 'jpg', 'jpeg', 'webp', 'mp4', 'webm', 'txt', 'json', 'md',
  'docx', 'pptx', 'epub', 'csv', 'ttf', 'html', 'svg', 'pdf', 'xlsx', 'zip', 'fig',
]);

interface SaveHost {
  save(request: { filename: string; data: string | Blob | ArrayBuffer | ArrayBufferView }): Promise<unknown>;
}

interface ClaudeHost {
  use(name: string): Promise<SaveHost | null>;
}

function host(): ClaudeHost | null {
  const w = window as unknown as { claude?: { use?: unknown } };
  return typeof w.claude?.use === 'function' ? (w.claude as ClaudeHost) : null;
}

/** True when this copy of the app runs inside a host that saves files for it. */
export function usesHostSave() {
  return host() !== null;
}

function extensionOf(filename: string) {
  return filename.slice(filename.lastIndexOf('.') + 1).toLowerCase();
}

function anchorSave(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Save `data` as `filename`. `wrap` turns the file into a single-entry ZIP,
 * which is how an STL reaches the user from a host that will not take one;
 * it is only called when that is needed, so the ZIP code stays out of the
 * ordinary path.
 */
export function saveFile(
  data: Uint8Array | string,
  filename: string,
  mime: string,
  wrap?: (name: string, data: Uint8Array | string) => { filename: string; bytes: Uint8Array },
) {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const blob = new Blob([bytes.slice().buffer], { type: mime });
  const h = host();
  if (!h) {
    anchorSave(blob, filename);
    return;
  }
  void (async () => {
    const notify = useUi.getState().notify;
    try {
      const downloads = await h.use('downloads');
      if (!downloads) {
        notify(tr('This copy of Organa cannot save files – run it locally to export'), 'warn');
        return;
      }
      if (HOST_EXTENSIONS.has(extensionOf(filename)) || !wrap) {
        await downloads.save({ filename, data: blob });
        return;
      }
      // An STL has to travel inside a ZIP here.
      const zipped = wrap(filename, bytes);
      await downloads.save({ filename: zipped.filename, data: new Blob([zipped.bytes.slice().buffer]) });
      notify(tr('Saved as {name} – this viewer only accepts a few file types', { name: zipped.filename }));
    } catch (e) {
      const code = (e as { code?: string })?.code;
      if (code === 'declined') return;
      notify(tr('The file could not be saved'), 'warn');
    }
  })();
}
