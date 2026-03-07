import { useColorScheme } from "react-native";

export const lightTheme = {
  bg: "#FAFAFE",
  surface: "#F2F0FD",
  card: "#FFFFFF",
  cardHighlight: "#EDEAFB",
  border: "#E4E2F8",

  accent: "#5E50E8",
  accentDim: "#5E50E80F",
  accentGlow: "#5E50E81C",

  green: "#059669",
  greenDim: "#05966912",

  orange: "#EA7215",

  text: "#14123A",
  textMuted: "#6258A0",
  textSub: "#9B96C8",

  purple: "#7C3AED",
  gold: "#D97706",
} as const;

export const darkTheme = {
  bg: "#0B0B12",
  surface: "#12121C",
  card: "#16161F",
  cardHighlight: "#1C1C2A",
  border: "#23233A",

  accent: "#7C6FFF",
  accentDim: "#7C6FFF14",
  accentGlow: "#7C6FFF28",

  green: "#34D399",
  greenDim: "#34D39914",

  orange: "#FB923C",

  text: "#ECEBFF",
  textMuted: "#7878A0",
  textSub: "#404060",

  purple: "#C4B5FD",
  gold: "#FCD34D",
} as const;

export function useTheme() {
  const colorScheme = useColorScheme();
  return colorScheme === "dark" ? darkTheme : lightTheme;
}

export type AppTheme = typeof lightTheme;
export type ThemeColor = (typeof lightTheme)[keyof typeof lightTheme];
