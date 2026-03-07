/**
 * ElevenLabs Conversational AI Integration
 *
 * Voice Interface for TalkBot
 *
 * Flow:
 * User speaks → ElevenLabs (STT) → This endpoint → TalkBot (Gemini + Semantic Search) → Response → ElevenLabs (TTS) → User hears
 *
 * This is just a thin wrapper around existing TalkBot service!
 * 100% code reuse - just different input/output format.
 */

import { Router, Request, Response } from 'express';
import { processChatMessage, generateSessionId } from '../services/talkbot.js';

const router = Router();

/**
 * POST /api/elevenlabs/webhook
 *
 * Called by ElevenLabs Conversational AI with user's transcribed speech
 *
 * Request body from ElevenLabs:
 * {
 *   "text": "I need someone who knows blockchain",
 *   "conversation_id": "conv-123-abc",
 *   "user_id": "user-456",
 *   "metadata": {}
 * }
 *
 * Response to ElevenLabs:
 * {
 *   "response": "I found Raj Kumar from DevFest - he's a blockchain expert!",
 *   "end_conversation": false
 * }
 */
router.post('/webhook', async (req: Request, res: Response) => {
  try {
    const {
      text,
      conversation_id,
      user_id,
      metadata
    } = req.body;

    console.log('[ELEVENLABS] Voice message received');
    console.log(`  User: ${user_id}`);
    console.log(`  Conversation: ${conversation_id}`);
    console.log(`  Transcribed text: "${text}"`);

    // Validation
    if (!text || typeof text !== 'string') {
      return res.status(400).json({
        error: 'text is required',
        response: 'Sorry, I didn\'t catch that. Could you repeat?',
        end_conversation: false
      });
    }

    if (!user_id) {
      return res.status(400).json({
        error: 'user_id is required',
        response: 'I need to know who you are to search your network.',
        end_conversation: true
      });
    }

    // Use existing TalkBot service! (100% code reuse)
    const sessionId = conversation_id || generateSessionId();
    const result = await processChatMessage(user_id, sessionId, text);

    if (!result.success) {
      console.error('[ELEVENLABS] TalkBot error:', result.error);
      return res.json({
        response: 'Sorry, I encountered an error. Please try again.',
        end_conversation: false
      });
    }

    // Log tool usage for debugging
    if (result.toolCalls && result.toolCalls.length > 0) {
      console.log('[ELEVENLABS] AI used tools:');
      result.toolCalls.forEach((call: any) => {
        console.log(`  - ${call.name}(${JSON.stringify(call.args)})`);
      });
    }

    console.log(`[ELEVENLABS] Response: "${result.response}"`);

    // Return response for ElevenLabs to speak
    res.json({
      response: result.response,
      end_conversation: false,  // Keep conversation going
      metadata: {
        toolCalls: result.toolCalls || [],
        sessionId: sessionId
      }
    });

  } catch (error: any) {
    console.error('[ELEVENLABS] Error processing voice message:', error);

    // Return graceful error response
    res.json({
      response: 'Sorry, something went wrong. Let\'s try that again.',
      end_conversation: false
    });
  }
});

/**
 * POST /api/elevenlabs/end-conversation
 *
 * Called when user ends the voice conversation
 *
 * Optional: Clean up resources, save final state
 */
router.post('/end-conversation', async (req: Request, res: Response) => {
  try {
    const { conversation_id, user_id } = req.body;

    console.log('[ELEVENLABS] Conversation ended');
    console.log(`  User: ${user_id}`);
    console.log(`  Conversation: ${conversation_id}`);

    // Optional: Save conversation summary, analytics, etc.

    res.json({
      success: true,
      message: 'Conversation ended successfully'
    });

  } catch (error: any) {
    console.error('[ELEVENLABS] Error ending conversation:', error);
    res.status(500).json({
      error: 'Failed to end conversation',
      details: error.message
    });
  }
});

/**
 * GET /api/elevenlabs/health
 *
 * Health check endpoint for ElevenLabs to verify webhook is reachable
 */
router.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    service: 'ElevenLabs Voice TalkBot',
    timestamp: new Date().toISOString()
  });
});

export default router;
