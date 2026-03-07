/**
 * Contact CRUD Routes
 *
 * Handles standard CRUD operations for contacts:
 * - List contacts (with filtering/sorting)
 * - Get single contact
 * - Update contact
 * - Delete contact
 */

import { Router, Request, Response } from 'express';
import { db } from '../db/index.js';
import { contacts, conversations } from '../db/schema.js';
import { eq, desc, sql, and } from 'drizzle-orm';
import { requireAuth, getUserId } from '../middleware/auth.js';

const router = Router();

/**
 * GET /api/contacts
 *
 * List all contacts for authenticated user
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * Query parameters:
 * - limit (optional): Number of contacts to return (default: 50)
 * - offset (optional): Pagination offset (default: 0)
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 */
router.get('/', requireAuth, async (req: Request, res: Response) => {
  try {
    const { limit = '50', offset = '0' } = req.query;

    // Get user ID from authenticated token
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[CONTACTS] Listing contacts for user: ${userId}`);

    const userContacts = await db
      .select()
      .from(contacts)
      .where(eq(contacts.userId, userId))
      .orderBy(desc(contacts.createdAt))
      .limit(parseInt(limit as string))
      .offset(parseInt(offset as string));

    console.log(`[CONTACTS] Found ${userContacts.length} contacts`);

    res.json({
      contacts: userContacts,
      count: userContacts.length,
      pagination: {
        limit: parseInt(limit as string),
        offset: parseInt(offset as string)
      }
    });

  } catch (error: any) {
    console.error('[CONTACTS] List error:', error);
    res.status(500).json({ error: 'Failed to fetch contacts' });
  }
});

/**
 * GET /api/contacts/:id
 *
 * Get a single contact by ID
 * Includes associated conversations
 *
 * PROTECTED: Requires valid Auth0 JWT token
 * User can only access their own contacts
 */
router.get('/:id', requireAuth, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[CONTACTS] Fetching contact: ${id}`);

    // Fetch contact (with ownership check)
    const [contact] = await db
      .select()
      .from(contacts)
      .where(and(
        eq(contacts.id, id),
        eq(contacts.userId, userId)  // Only fetch if user owns it
      ));

    if (!contact) {
      return res.status(404).json({ error: 'Contact not found' });
    }

    // Fetch associated conversations
    const contactConversations = await db
      .select()
      .from(conversations)
      .where(eq(conversations.contactId, id))
      .orderBy(desc(conversations.createdAt));

    console.log(`[CONTACTS] Contact found with ${contactConversations.length} conversations`);

    res.json({
      contact,
      conversations: contactConversations
    });

  } catch (error: any) {
    console.error('[CONTACTS] Get error:', error);
    res.status(500).json({ error: 'Failed to fetch contact' });
  }
});

/**
 * PUT /api/contacts/:id
 *
 * Update a contact
 *
 * PROTECTED: Requires valid Auth0 JWT token
 * User can only update their own contacts
 *
 * Request body: Any contact fields to update
 * {
 *   "name": "Updated Name",
 *   "phone": "+1234567890",
 *   "notes": "Follow up next week",
 *   ...
 * }
 */
router.put('/:id', requireAuth, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const updates = req.body;
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[CONTACTS] Updating contact: ${id}`);

    // Add updatedAt timestamp
    const updateData = {
      ...updates,
      updatedAt: new Date()
    };

    // Update only if user owns the contact
    const [updated] = await db
      .update(contacts)
      .set(updateData)
      .where(and(
        eq(contacts.id, id),
        eq(contacts.userId, userId)
      ))
      .returning();

    if (!updated) {
      return res.status(404).json({ error: 'Contact not found' });
    }

    console.log(`[CONTACTS] Contact updated successfully`);

    res.json({
      success: true,
      contact: updated
    });

  } catch (error: any) {
    console.error('[CONTACTS] Update error:', error);
    res.status(500).json({ error: 'Failed to update contact' });
  }
});

/**
 * DELETE /api/contacts/:id
 *
 * Delete a contact (and all associated conversations)
 * This is a CASCADE delete - conversations will also be deleted
 *
 * PROTECTED: Requires valid Auth0 JWT token
 * User can only delete their own contacts
 */
router.delete('/:id', requireAuth, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[CONTACTS] Deleting contact: ${id}`);

    // Delete only if user owns the contact
    const deleted = await db
      .delete(contacts)
      .where(and(
        eq(contacts.id, id),
        eq(contacts.userId, userId)
      ))
      .returning();

    if (deleted.length === 0) {
      return res.status(404).json({ error: 'Contact not found' });
    }

    console.log(`[CONTACTS] Contact deleted successfully`);

    res.json({
      success: true,
      message: 'Contact deleted successfully'
    });

  } catch (error: any) {
    console.error('[CONTACTS] Delete error:', error);
    res.status(500).json({ error: 'Failed to delete contact' });
  }
});

/**
 * GET /api/contacts/search
 *
 * Search contacts by name, company, or topics
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * Query parameters:
 * - query (required): Search query
 */
router.get('/search', requireAuth, async (req: Request, res: Response) => {
  try {
    const { query } = req.query;
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'query parameter is required' });
    }

    console.log(`[CONTACTS] Searching for: "${query}"`);

    // Search in name, company, role, and conversation summary
    const searchResults = await db
      .select()
      .from(contacts)
      .where(
        sql`
          ${contacts.userId} = ${userId}
          AND (
            ${contacts.name} ILIKE ${'%' + query + '%'}
            OR ${contacts.company} ILIKE ${'%' + query + '%'}
            OR ${contacts.role} ILIKE ${'%' + query + '%'}
            OR ${contacts.conversationSummary} ILIKE ${'%' + query + '%'}
          )
        `
      )
      .orderBy(desc(contacts.createdAt))
      .limit(20);

    console.log(`[CONTACTS] Found ${searchResults.length} matching contacts`);

    res.json({
      contacts: searchResults,
      count: searchResults.length,
      query
    });

  } catch (error: any) {
    console.error('[CONTACTS] Search error:', error);
    res.status(500).json({ error: 'Failed to search contacts' });
  }
});

export default router;
