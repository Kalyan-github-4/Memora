export type ContactStatus = "hot" | "warm" | "cold";

export interface Contact {
  id: number;
  name: string;
  role: string;
  company: string;
  location: string;
  avatar: string;
  color: string;
  topics: string[];
  followUps: string[];
  contactInfo: string[];
  summary: string;
  metAt: string;
  metDate: string;
  img: string;
}

export type ExperienceLevel = "<1yr" | "1-3yrs" | "3-5yrs" | "5-10yrs" | "10+yrs";

export type NetworkingGoal =
  | "hiring"
  | "job-seeking"
  | "partnerships"
  | "fundraising"
  | "learning"
  | "mentoring"
  | "networking"
  | "clients";

export type FollowUpStyle = "quick" | "balanced" | "detailed";
export type ConversationStyle = "casual" | "professional" | "mixed";

export interface UserProfile {
  // Identity
  name: string;
  role: string;
  company: string;
  location: string;
  bio: string;
  // Professional
  industry: string;
  experience: ExperienceLevel;
  expertise: string[];
  // Networking
  networkingGoals: NetworkingGoal[];
  // Interests
  interests: string[];
  // Preferences
  followUpStyle: FollowUpStyle;
  conversationStyle: ConversationStyle;
}

export interface AuthUser {
  name: string;
  email: string;
}

export interface AuthContextType {
  user: AuthUser | null;
  isLoading: boolean;
  signIn: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  signUp: (name: string, email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  signOut: () => Promise<void>;
}
