/**
 * TalkBot Chat Routes
 *
 * REST API endpoints for TalkBot (complement to Socket.io)
 *
 * THE CORE FEATURE:
 * - User discusses ideas with AI
 * - AI helps find collaborators from their network
 * - Semantic search powered by vector embeddings
 */

import { Router, Request, Response } from 'express';
import { requireAuth, getUserId } from '../middleware/auth.js';
import { processChatMessage, generateSessionId, getChatSession } from '../services/talkbot.js';
import { generateMissingEmbeddings } from '../services/semanticSearch.js';
import { emitToUser } from '../services/socket.js';

const router = Router();

/**
 * POST /api/chat/message
 *
 * Send a message to TalkBot
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * Body:
 * {
 *   "message": "I want to build a blockchain payment system",
 *   "sessionId": "session-123" // optional, will create new if not provided
 * }
 */
router.post('/message', requireAuth, async (req: Request, res: Response) => {
  try {
    const { message, sessionId } = req.body;
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }

    // Create session ID if not provided
    const activeSessionId = sessionId || generateSessionId();

    console.log(`[CHAT] Message from user ${userId}`);
    console.log(`  Session: ${activeSessionId}`);
    console.log(`  Message: "${message}"`);

    // Emit typing indicator via Socket.io
    try {
      emitToUser(userId, 'chat:typing', { isTyping: true });
    } catch (error) {
      console.warn('[CHAT] Failed to emit typing indicator (Socket.io might not be connected)');
    }

    // Process message with AI
    const result = await processChatMessage(userId, activeSessionId, message);

    // Stop typing indicator
    try {
      emitToUser(userId, 'chat:typing', { isTyping: false });
    } catch (error) {
      // Ignore
    }

    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to process message',
        details: result.error
      });
    }

    // Send response via Socket.io (if connected)
    try {
      emitToUser(userId, 'chat:message', {
        role: 'assistant',
        content: result.response,
        toolCalls: result.toolCalls,
        sessionId: activeSessionId
      });
    } catch (error) {
      console.warn('[CHAT] Failed to emit via Socket.io');
    }

    // Return response via HTTP
    res.json({
      success: true,
      sessionId: activeSessionId,
      response: result.response,
      toolCalls: result.toolCalls
    });

  } catch (error: any) {
    console.error('[CHAT] Error processing message:', error);
    res.status(500).json({
      error: 'Failed to process message',
      details: error.message
    });
  }
});

/**
 * GET /api/chat/session/:sessionId
 *
 * Get chat history for a session
 *
 * PROTECTED: Requires valid Auth0 JWT token
 */
router.get('/session/:sessionId', requireAuth, async (req: Request, res: Response) => {
  try {
    const { sessionId } = req.params;
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[CHAT] Fetching session ${sessionId} for user ${userId}`);

    const messages = await getChatSession(userId, sessionId);

    res.json({
      success: true,
      sessionId,
      messages: messages.map(msg => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        toolCalls: msg.toolCalls,
        toolResults: msg.toolResults,
        createdAt: msg.createdAt
      }))
    });

  } catch (error: any) {
    console.error('[CHAT] Error fetching session:', error);
    res.status(500).json({
      error: 'Failed to fetch session',
      details: error.message
    });
  }
});

/**
 * POST /api/chat/generate-embeddings
 *
 * Manually trigger embedding generation for all contacts
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * This is useful when:
 * - User has existing contacts without embeddings
 * - Want to regenerate embeddings after changes
 */
router.post('/generate-embeddings', requireAuth, async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log(`[CHAT] Generating embeddings for user ${userId}`);

    // Start generation (async)
    generateMissingEmbeddings(userId).catch(error => {
      console.error('[CHAT] Error generating embeddings:', error);
    });

    res.json({
      success: true,
      message: 'Embedding generation started. This may take a few minutes.'
    });

  } catch (error: any) {
    console.error('[CHAT] Error starting embedding generation:', error);
    res.status(500).json({
      error: 'Failed to start embedding generation',
      details: error.message
    });
  }
});

export default router;
