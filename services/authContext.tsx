import * as SecureStore from "expo-secure-store";
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { AuthUser, AuthContextType } from "@/types";
const AUTH_KEY = "memora_auth_user";

const AuthContext = createContext<AuthContextType>({
  user: null,
  isLoading: true,
  signIn: async () => ({ success: false }),
  signUp: async () => ({ success: false }),
  signOut: async () => {},
});

export function AuthProvider({ children }: Readonly<{ children: React.ReactNode }>) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    SecureStore.getItemAsync(AUTH_KEY)
      .then((raw) => {
        if (raw) setUser(JSON.parse(raw));
      })
      .catch(() => {})
      .finally(() => setIsLoading(false));
  }, []);

  const signUp = useCallback(async (name: string, email: string, _password: string) => {
    if (!name.trim() || !email.trim() || !_password.trim()) {
      return { success: false, error: "All fields are required" };
    }
    const authUser: AuthUser = { name: name.trim(), email: email.trim().toLowerCase() };
    await SecureStore.setItemAsync(AUTH_KEY, JSON.stringify(authUser));
    setUser(authUser);
    return { success: true };
  }, []);

  const signIn = useCallback(async (email: string, _password: string) => {
    if (!email.trim() || !_password.trim()) {
      return { success: false, error: "All fields are required" };
    }
    // Check if a user was previously registered
    const raw = await SecureStore.getItemAsync(AUTH_KEY);
    if (raw) {
      const stored: AuthUser = JSON.parse(raw);
      if (stored.email === email.trim().toLowerCase()) {
        setUser(stored);
        return { success: true };
      }
    }
    // For demo: allow any sign-in and create user
    const authUser: AuthUser = { name: email.split("@")[0], email: email.trim().toLowerCase() };
    await SecureStore.setItemAsync(AUTH_KEY, JSON.stringify(authUser));
    setUser(authUser);
    return { success: true };
  }, []);

  const signOut = useCallback(async () => {
    await SecureStore.deleteItemAsync(AUTH_KEY);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, isLoading, signIn, signUp, signOut }),
    [user, isLoading, signIn, signUp, signOut],
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
