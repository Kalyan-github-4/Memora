/**
 * Database Migration Script
 *
 * This script creates the database tables if they don't exist.
 * Run with: npx tsx src/db/migrate.ts
 */

import postgres from 'postgres';
import * as dotenv from 'dotenv';

dotenv.config();

const connectionString = process.env.DATABASE_URL!;

if (!connectionString) {
  throw new Error('DATABASE_URL environment variable is not set');
}

const sql = postgres(connectionString);

async function migrate() {
  console.log('[DB] Starting database migration...');

  try {
    // Create users table
    await sql`
      CREATE TABLE IF NOT EXISTS "users" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
        "email" text NOT NULL,
        "name" text NOT NULL,
        "created_at" timestamp DEFAULT now() NOT NULL,
        CONSTRAINT "users_email_unique" UNIQUE("email")
      )
    `;
    console.log('[DB] ✓ Users table created/verified');

    // Create contacts table
    await sql`
      CREATE TABLE IF NOT EXISTS "contacts" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
        "user_id" uuid NOT NULL,
        "name" text,
        "role" text,
        "company" text,
        "location" text,
        "phone" text,
        "email" text,
        "linkedin_url" text,
        "topics_discussed" text[],
        "follow_ups" text[],
        "conversation_summary" text,
        "confidence_score" real,
        "met_at" text,
        "met_date" timestamp,
        "met_location" text,
        "created_at" timestamp DEFAULT now() NOT NULL,
        "updated_at" timestamp DEFAULT now() NOT NULL
      )
    `;
    console.log('[DB] ✓ Contacts table created/verified');

    // Create conversations table
    await sql`
      CREATE TABLE IF NOT EXISTS "conversations" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
        "contact_id" uuid NOT NULL,
        "audio_url" text NOT NULL,
        "raw_transcription" text,
        "diarized_transcription" jsonb,
        "conversation_summary" text,
        "duration_seconds" real,
        "event_name" text,
        "created_at" timestamp DEFAULT now() NOT NULL
      )
    `;
    console.log('[DB] ✓ Conversations table created/verified');

    // Add foreign key constraints (with error handling for duplicates)
    try {
      await sql`
        ALTER TABLE "contacts"
        ADD CONSTRAINT "contacts_user_id_users_id_fk"
        FOREIGN KEY ("user_id") REFERENCES "users"("id")
        ON DELETE cascade
        ON UPDATE no action
      `;
      console.log('[DB] ✓ Contacts foreign key constraint added');
    } catch (e: any) {
      if (e.message.includes('already exists')) {
        console.log('[DB] ✓ Contacts foreign key constraint already exists');
      } else {
        throw e;
      }
    }

    try {
      await sql`
        ALTER TABLE "conversations"
        ADD CONSTRAINT "conversations_contact_id_contacts_id_fk"
        FOREIGN KEY ("contact_id") REFERENCES "contacts"("id")
        ON DELETE cascade
        ON UPDATE no action
      `;
      console.log('[DB] ✓ Conversations foreign key constraint added');
    } catch (e: any) {
      if (e.message.includes('already exists')) {
        console.log('[DB] ✓ Conversations foreign key constraint already exists');
      } else {
        throw e;
      }
    }

    console.log('[DB] Migration completed successfully!');
  } catch (error) {
    console.error('[DB] Migration failed:', error);
    throw error;
  } finally {
    await sql.end();
  }
}

// Run migration
migrate().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
