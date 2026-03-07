/**
 * Audio Processing Routes
 *
 * Handles audio processing from URLs (remote files)
 * Uses shared audioProcessor service for common logic
 */

import { Router, Request, Response } from 'express';
import { processAndStoreContact } from '../services/audioProcessor.js';
import { requireAuth, getUserId } from '../middleware/auth.js';

const router = Router();

/**
 * POST /api/audio/process
 *
 * Process audio file and extract contact information
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * Request body:
 * {
 *   "audioUrl": "https://storage.supabase.co/bucket/audio.wav",
 *   "eventContext": {
 *     "event_name": "DevFest Kolkata 2026",
 *     "location": "Kolkata, India",
 *     "timestamp": "2026-03-04T15:30:00Z"
 *   }
 * }
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 */
router.post('/process', requireAuth, async (req: Request, res: Response) => {
  try {
    const { audioUrl, eventContext } = req.body;

    // Get user ID from authenticated token
    const userId = getUserId(req);

    if (!userId) {
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    // Validate required fields
    if (!audioUrl) {
      return res.status(400).json({ error: 'audioUrl is required' });
    }

    console.log('[AUDIO] Processing request from URL');
    console.log(`  Audio URL: ${audioUrl}`);
    console.log(`  User ID: ${userId}`);

    // Use shared processor service
    const result = await processAndStoreContact({
      audioPath: audioUrl,
      userId: userId,
      eventContext: eventContext
    });

    // Handle result
    if (!result.success) {
      // Determine appropriate status code
      let statusCode = 500;
      if (result.error === 'Python AI service is not available') {
        statusCode = 503;
      }

      return res.status(statusCode).json({
        error: result.error,
        details: result.details
      });
    }

    // Success response
    res.json({
      success: true,
      contact: result.contact,
      metadata: result.metadata
    });

  } catch (error: any) {
    console.error('[AUDIO] Unexpected error:', error.message);
    res.status(500).json({
      error: 'Failed to process audio',
      details: error.message
    });
  }
});

export default router;
