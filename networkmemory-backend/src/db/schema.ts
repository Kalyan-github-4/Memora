/**
 * NetworkMemory Database Schema (Drizzle ORM)
 *
 * This is the SINGLE SOURCE OF TRUTH for the database schema.
 * Python AI service does NOT access the database - it's stateless.
 * Only Node.js backend writes to the database.
 */

import { pgTable, uuid, text, timestamp, real, jsonb, customType } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';

// Define custom vector type for pgvector
const vector = customType<{ data: number[]; driverData: string }>({
  dataType() {
    return 'vector(768)'; // 768 dimensions for Gemini embeddings
  },
  toDriver(value: number[]): string {
    return JSON.stringify(value);
  },
  fromDriver(value: string): number[] {
    return JSON.parse(value);
  }
});

/**
 * Users table
 * Stores user accounts and authentication info
 */

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

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: text('email').notNull().unique(),
  name: text('name').notNull(),

  // Professional Information
  role: text('role'),
  company: text('company'),
  location: text('location'),
  bio: text('bio'),
  industry: text('industry'),
  experience: text('experience').$type<ExperienceLevel>(),
  expertise: text('expertise').array(),

  // Networking
  networkingGoals: text('networking_goals').array().$type<NetworkingGoal[]>(),

  // Interests
  interests: text('interests').array(),

  // Preferences
  followUpStyle: text('follow_up_style').$type<FollowUpStyle>(),
  conversationStyle: text('conversation_style').$type<ConversationStyle>(),

  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull()
});

/**
 * Contacts table
 * Stores extracted contact information from networking conversations
 *
 * Why these fields:
 * - Basic info: name, role, company, location (who they are)
 * - Contact methods: phone, email, linkedin_url (how to reach them)
 * - Conversation data: topics_discussed, follow_ups (what was discussed)
 * - Metadata: confidence_score (how confident is the extraction)
 * - Event context: met_at, met_date, met_location (where you met)
 */
export const contacts = pgTable('contacts', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  // Basic contact information
  name: text('name'),
  role: text('role'),
  company: text('company'),
  location: text('location'),
  phone: text('phone'),
  email: text('email'),
  linkedinUrl: text('linkedin_url'),

  // Extracted conversation data
  topicsDiscussed: text('topics_discussed').array(),
  followUps: text('follow_ups').array(),
  conversationSummary: text('conversation_summary'),
  confidenceScore: real('confidence_score'),

  // Event context
  metAt: text('met_at'),
  metDate: timestamp('met_date', { mode: 'date' }),
  metLocation: text('met_location'),

  // Semantic search (pgvector - 768 dimensions for Gemini embeddings)
  embedding: vector('embedding'),

  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull()
});

/**
 * Conversations table
 * Stores raw conversation data linked to contacts
 *
 * Why separate from contacts:
 * - One contact might have multiple conversations over time
 * - Useful for tracking conversation history
 * - Can re-process conversations later if needed
 */
export const conversations = pgTable('conversations', {
  id: uuid('id').primaryKey().defaultRandom(),
  contactId: uuid('contact_id').notNull().references(() => contacts.id, { onDelete: 'cascade' }),

  // Audio and transcription data
  audioUrl: text('audio_url').notNull(),
  rawTranscription: text('raw_transcription'),
  diarizedTranscription: jsonb('diarized_transcription'),  // Array of utterances with speaker labels

  // Summary and metadata
  conversationSummary: text('conversation_summary'),
  durationSeconds: real('duration_seconds'),
  eventName: text('event_name'),

  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull()
});

/**
 * Chat Messages table (TalkBot)
 * Stores TalkBot conversation history
 *
 * THE CORE FEATURE: User discusses ideas with AI, AI helps find collaborators
 */
export const chatMessages = pgTable('chat_messages', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  // Message data
  role: text('role').notNull(),  // 'user' or 'assistant'
  content: text('content').notNull(),

  // Tool usage tracking (when AI searches for contacts)
  toolCalls: jsonb('tool_calls'),  // Array of tool calls made
  toolResults: jsonb('tool_results'),  // Array of tool results

  // Context tracking
  sessionId: text('session_id'),  // Group messages into sessions

  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull()
});

// Export types for use in the application
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

export type Contact = typeof contacts.$inferSelect;
export type NewContact = typeof contacts.$inferInsert;

export type Conversation = typeof conversations.$inferSelect;
export type NewConversation = typeof conversations.$inferInsert;

export type ChatMessage = typeof chatMessages.$inferSelect;
export type NewChatMessage = typeof chatMessages.$inferInsert;
