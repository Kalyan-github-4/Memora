import { Contact } from "@/types";
import { calculateContactScore } from "./contactScore";

export function contactStrength(c: Contact): "hot" | "warm" | "cold" {
  const score = calculateContactScore(c);
  if (score >= 80) return "hot";
  if (score >= 50) return "warm";
  return "cold";
}
