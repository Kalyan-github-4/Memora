/**
 * Semantic Search Service (with pgvector)
 *
 * Searches contacts using vector embeddings and PostgreSQL pgvector extension
 *
 * Flow:
 * 1. When contact is created → generate embedding from (topics + role + company)
 * 2. When user searches → generate embedding from search query
 * 3. Compare embeddings using pgvector's <-> operator (cosine distance)
 * 4. Return most similar contacts
 *
 * PERFORMANCE:
 * - pgvector is 10-100x faster than manual cosine similarity
 * - Can handle millions of vectors with indexes
 */

import { GoogleGenAI } from '@google/genai';
import { db } from '../db/index.js';
import { contacts } from '../db/schema.js';
import { eq, sql, and, isNotNull } from 'drizzle-orm';

// Initialize Gemini AI client with API key
const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

/**
 * Generate text embedding using Gemini
 *
 * Returns 768-dimensional vector
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  try {
    // Use new @google/genai library with gemini-embedding-001 model
    // Request 768 dimensions (database schema uses vector(768))
    const response = await genai.models.embedContent({
      model: 'gemini-embedding-001',
      contents: [text],
      config: {
        outputDimensionality: 768
      }
    });

    // Extract embedding values
    if (!response.embeddings || response.embeddings.length === 0) {
      throw new Error('No embeddings returned from API');
    }

    const values = response.embeddings[0].values;
    if (!values) {
      throw new Error('Embedding values are undefined');
    }

    return values;
  } catch (error: any) {
    console.error('[SEMANTIC] Error generating embedding:', error);
    throw new Error('Failed to generate embedding');
  }
}

/**
 * Generate and store embedding for a contact
 *
 * Uses pgvector to store as native vector type
 */
export async function generateContactEmbedding(contactId: string): Promise<void> {
  try {
    console.log(`[SEMANTIC] Generating embedding for contact ${contactId}`);

    // Fetch contact
    const [contact] = await db
      .select()
      .from(contacts)
      .where(eq(contacts.id, contactId));

    if (!contact) {
      throw new Error('Contact not found');
    }

    // Build text representation for embedding
    const textParts: string[] = [];

    if (contact.role) textParts.push(`Role: ${contact.role}`);
    if (contact.company) textParts.push(`Company: ${contact.company}`);
    if (contact.topicsDiscussed && contact.topicsDiscussed.length > 0) {
      textParts.push(`Topics: ${contact.topicsDiscussed.join(', ')}`);
    }
    if (contact.conversationSummary) {
      textParts.push(`Summary: ${contact.conversationSummary}`);
    }

    const textToEmbed = textParts.join('. ');

    if (!textToEmbed.trim()) {
      console.warn(`[SEMANTIC] No meaningful text for contact ${contactId}, skipping embedding`);
      return;
    }

    console.log(`[SEMANTIC] Text to embed: "${textToEmbed.substring(0, 100)}..."`);

    // Generate embedding (768 dimensions)
    const embedding = await generateEmbedding(textToEmbed);

    // Store embedding using pgvector
    // The custom vector type handles conversion automatically
    await db
      .update(contacts)
      .set({ embedding: embedding })
      .where(eq(contacts.id, contactId));

    console.log(`[SEMANTIC] Embedding stored for contact ${contactId}`);

  } catch (error: any) {
    console.error('[SEMANTIC] Error generating contact embedding:', error);
    throw error;
  }
}

/**
 * Search contacts semantically using pgvector
 *
 * @param userId - User ID
 * @param query - Search query (e.g., "blockchain expert", "marketing specialist")
 * @param limit - Maximum number of results
 * @returns Array of contacts sorted by similarity
 */
export async function searchContactsSemantically(
  userId: string,
  query: string,
  limit: number = 5
): Promise<any[]> {
  try {
    console.log(`[SEMANTIC] Searching contacts for user ${userId}`);
    console.log(`  Query: "${query}"`);

    // Generate embedding for search query
    const queryEmbedding = await generateEmbedding(query);

    // Convert to PostgreSQL vector string format
    const vectorString = `[${queryEmbedding.join(',')}]`;

    // Use pgvector's <-> operator for cosine distance
    // Note: <-> returns distance (0 = identical, higher = less similar)
    // So we calculate similarity as (1 - distance)
    const results = await db
      .select({
        id: contacts.id,
        userId: contacts.userId,
        name: contacts.name,
        role: contacts.role,
        company: contacts.company,
        location: contacts.location,
        email: contacts.email,
        phone: contacts.phone,
        linkedinUrl: contacts.linkedinUrl,
        topicsDiscussed: contacts.topicsDiscussed,
        followUps: contacts.followUps,
        conversationSummary: contacts.conversationSummary,
        confidenceScore: contacts.confidenceScore,
        metAt: contacts.metAt,
        metDate: contacts.metDate,
        metLocation: contacts.metLocation,
        createdAt: contacts.createdAt,
        updatedAt: contacts.updatedAt,
        // Calculate similarity: 1 - distance = similarity
        similarity: sql<number>`1 - (embedding <-> ${sql.raw(`'${vectorString}'::vector`)})`
      })
      .from(contacts)
      .where(
        and(
          eq(contacts.userId, userId),
          isNotNull(contacts.embedding)  // Only search contacts with embeddings
        )
      )
      .orderBy(sql`embedding <-> ${sql.raw(`'${vectorString}'::vector`)}`)  // Order by distance (ascending)
      .limit(limit);

    console.log(`[SEMANTIC] Found ${results.length} matching contacts`);
    results.forEach((c, i) => {
      console.log(`  ${i + 1}. ${c.name} (${c.role} at ${c.company}) - Similarity: ${((c.similarity || 0) * 100).toFixed(1)}%`);
    });

    return results;

  } catch (error: any) {
    console.error('[SEMANTIC] Error searching contacts:', error);

    // If pgvector is not enabled, provide helpful error
    if (error.message?.includes('type "vector" does not exist')) {
      console.error('\n❌ pgvector extension is not enabled!');
      console.error('Run this in your Neon DB SQL Editor:');
      console.error('  CREATE EXTENSION IF NOT EXISTS vector;\n');
    }

    throw error;
  }
}

/**
 * Batch generate embeddings for all contacts without embeddings
 */
export async function generateMissingEmbeddings(userId: string): Promise<void> {
  try {
    console.log(`[SEMANTIC] Generating missing embeddings for user ${userId}`);

    // Find contacts without embeddings
    const contactsWithoutEmbeddings = await db
      .select()
      .from(contacts)
      .where(
        and(
          eq(contacts.userId, userId),
          sql`${contacts.embedding} IS NULL`
        )
      );

    console.log(`[SEMANTIC] Found ${contactsWithoutEmbeddings.length} contacts without embeddings`);

    for (const contact of contactsWithoutEmbeddings) {
      await generateContactEmbedding(contact.id);
    }

    console.log(`[SEMANTIC] Finished generating embeddings`);

  } catch (error: any) {
    console.error('[SEMANTIC] Error generating missing embeddings:', error);
    throw error;
  }
}
