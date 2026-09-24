// localStorage can be missing or throw (private windows, blocked site data); never let that break the app.

export const THEME_KEY = 'headroom.theme';
export const TOKEN_KEY = 'headroom.hfToken';
export const REMEMBER_TOKEN_KEY = 'headroom.rememberToken';

export function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string | null): void {
  try {
    if (value === null || value === '') window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // ignore: persistence is a convenience
  }
}
