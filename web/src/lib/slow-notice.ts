/**
 * Tracks requests to the API and says when any has been waiting a while. On a free host the first request after a
 * quiet spell waits for the server to wake up (up to a minute); without a hint that looks like a broken site.
 */
const SLOW_AFTER_MS = 4000;

let pending = 0;
let slow = false;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function trackRequest<T>(work: Promise<T>): Promise<T> {
  pending += 1;
  const timer = setTimeout(() => {
    if (pending > 0 && !slow) { slow = true; emit(); }
  }, SLOW_AFTER_MS);
  return work.finally(() => {
    clearTimeout(timer);
    pending -= 1;
    if (pending === 0 && slow) { slow = false; emit(); }
  });
}

export const subscribeSlow = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
export const isSlow = () => slow;
