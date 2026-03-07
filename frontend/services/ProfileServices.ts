import type { UserProfile } from "@/types";

let profile: UserProfile = {
  name: "",
  role: "",
  company: "",
  location: "",
  bio: "",
  industry: "",
  experience: "1-3yrs",
  expertise: [],
  networkingGoals: [],
  interests: [],
  followUpStyle: "balanced",
  conversationStyle: "mixed",
};

export function getProfile() {
  return profile;
}

export function updateProfile(data: Partial<UserProfile>) {
  profile = { ...profile, ...data };
  return profile;
}
