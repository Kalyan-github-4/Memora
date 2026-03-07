import { Contact } from "../types";

export function calculateContactScore(c: Contact): number {
  let score = 0;

  // Identity (25)
  if (c.name) score += 5;
  if (c.role) score += 8;
  if (c.company) score += 8;
  if (c.location) score += 4;

  // Contact Depth (20)
  if (c.contactInfo?.length) {
    score += Math.min(c.contactInfo.length * 5, 20);
  }

  // Knowledge Depth (30)
  if (c.topics?.length) {
    score += Math.min(c.topics.length * 5, 15);
  }
  if (c.summary && c.summary.length > 40) {
    score += 15;
  }

  // Context + Relationship (25)
  if (c.metAt) score += 10;
  if (c.metDate) score += 5;
  if (c.followUps?.length) {
    score += Math.min(c.followUps.length * 5, 10);
  }

  return Math.min(score, 100);
}

export function contactScore(c: Contact): { score: number; color: string } {
  const score = calculateContactScore(c);

  let color = "#ef4444";
  if (score >= 80) color = "#34D399";
  else if (score >= 60) color = "#7C6FFF";
  else if (score >= 40) color = "#FB923C";
  else if (score >= 20) color = "#f97316";

  return { score, color };
}
