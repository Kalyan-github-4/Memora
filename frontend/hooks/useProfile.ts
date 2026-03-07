import type { UserProfile } from "@/types";
import { useAuth } from "@/services/authContext";
import * as SecureStore from "expo-secure-store";
import { useCallback, useEffect, useRef, useState } from "react";

const PROFILE_KEY = "memora_user_profile";

const DEFAULT_PROFILE: UserProfile = {
  name: "",
  role: "",
  company: "",
  location: "",
  bio: "",
  industry: "",
  experience: "3-5yrs",
  expertise: [],
  networkingGoals: [],
  interests: [],
  followUpStyle: "balanced",
  conversationStyle: "mixed",
};

export function useProfile() {
  const { user } = useAuth();
  const [profile, setProfile] = useState<UserProfile>(DEFAULT_PROFILE);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Track the last-saved version to compute isDirty accurately
  const savedRef = useRef<string>(JSON.stringify(DEFAULT_PROFILE));

  useEffect(() => {
    SecureStore.getItemAsync(PROFILE_KEY)
      .then((raw) => {
        if (raw) {
          const parsed: UserProfile = { ...DEFAULT_PROFILE, ...JSON.parse(raw) };
          setProfile(parsed);
          savedRef.current = JSON.stringify(parsed);
        } else if (user) {
          // First launch: seed profile from auth data
          const seeded: UserProfile = { ...DEFAULT_PROFILE, name: user.name };
          setProfile(seeded);
        }
      })
      .catch(() => {})
      .finally(() => setIsLoading(false));
  }, [user]);

  const update = useCallback((patch: Partial<UserProfile>) => {
    setProfile((prev) => {
      const next = { ...prev, ...patch };
      setIsDirty(JSON.stringify(next) !== savedRef.current);
      return next;
    });
  }, []);

  const save = useCallback(
    async (latest: UserProfile) => {
      setIsSaving(true);
      try {
        const raw = JSON.stringify(latest);
        await SecureStore.setItemAsync(PROFILE_KEY, raw);
        savedRef.current = raw;
        setIsDirty(false);
      } finally {
        setIsSaving(false);
      }
    },
    [],
  );

  return { profile, update, save, isDirty, isSaving, isLoading };
}
