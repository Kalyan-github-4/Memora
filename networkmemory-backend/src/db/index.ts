/**
 * Database Connection (Neon DB via Drizzle ORM)
 *
 * This module sets up the connection to Neon DB using Drizzle ORM.
 * All database operations go through this single connection.
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema.js';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Create PostgreSQL connection
const connectionString = process.env.DATABASE_URL!;

if (!connectionString) {
  throw new Error('DATABASE_URL environment variable is not set');
}

// Create postgres client
const client = postgres(connectionString);

// Create Drizzle instance
export const db = drizzle(client, { schema });

/**
 * Test database connection
 * Call this on server startup to ensure database is accessible
 */
export async function testConnection() {
  try {
    // Simple query to test connection
    const result = await client`SELECT 1 as connected`;
    console.log('[DB] Connected to Neon DB successfully');
    return true;
  } catch (error) {
    console.error('[DB] Failed to connect to database:', error);
    return false;
  }
}
