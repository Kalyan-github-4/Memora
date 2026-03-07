/**
 * File Upload Routes
 *
 * Handles audio file uploads using local storage
 * Files are stored in: uploads/audio/{userId}/{timestamp}-{filename}
 *
 * For hackathon: Using local storage (simple, fast)
 * For production: Can migrate to AWS S3/Cloudinary/Google Cloud Storage
 */

import { Router, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { processAndStoreContact } from '../services/audioProcessor.js';
import { requireAuth, getUserId } from '../middleware/auth.js';

const router = Router();

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Upload directory configuration (from environment or default)
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '../../uploads/audio');

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
  console.log('[UPLOAD] Created upload directory:', UPLOAD_DIR);
}

// Configure multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Create user-specific subdirectory
    const userId = req.body.userId || 'anonymous';
    const userDir = path.join(UPLOAD_DIR, userId);

    if (!fs.existsSync(userDir)) {
      fs.mkdirSync(userDir, { recursive: true });
    }

    cb(null, userDir);
  },
  filename: (req, file, cb) => {
    // Generate unique filename: timestamp-originalname
    const timestamp = Date.now();
    const sanitizedName = file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    const filename = `${timestamp}-${sanitizedName}`;
    cb(null, filename);
  }
});

// File filter - only allow audio files
const fileFilter = (req: any, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  const allowedMimes = [
    'audio/mpeg',      // mp3
    'audio/wav',       // wav
    'audio/wave',      // wav alternative
    'audio/x-wav',     // wav alternative
    'audio/mp4',       // m4a
    'audio/x-m4a',     // m4a alternative
    'audio/ogg',       // ogg
    'audio/webm',      // webm
    'video/mp4'        // sometimes m4a is detected as video/mp4
  ];

  if (allowedMimes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type. Allowed types: mp3, wav, m4a, ogg, webm. Got: ${file.mimetype}`));
  }
};

// Configure multer
const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 100 * 1024 * 1024  // 100MB max
  }
});

/**
 * POST /api/upload/audio
 *
 * Upload audio file
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * Form fields:
 * - audio: File (required) - Audio file to upload
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 *
 * Response:
 * {
 *   "success": true,
 *   "file": {
 *     "filename": "1709586234567-audio.wav",
 *     "path": "/uploads/audio/user-id/1709586234567-audio.wav",
 *     "size": 1234567,
 *     "mimetype": "audio/wav"
 *   }
 * }
 */
router.post('/audio', requireAuth, upload.single('audio'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    // Get user ID from authenticated token
    const userId = getUserId(req);

    if (!userId) {
      fs.unlinkSync(req.file.path);
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    console.log('[UPLOAD] Audio file uploaded successfully');
    console.log(`  User ID: ${userId}`);
    console.log(`  Filename: ${req.file.filename}`);
    console.log(`  Size: ${(req.file.size / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Path: ${req.file.path}`);

    res.json({
      success: true,
      file: {
        filename: req.file.filename,
        path: req.file.path,
        size: req.file.size,
        mimetype: req.file.mimetype,
        originalName: req.file.originalname
      }
    });

  } catch (error: any) {
    console.error('[UPLOAD] Upload error:', error);

    // Clean up file if error occurred
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      error: 'File upload failed',
      details: error.message
    });
  }
});

/**
 * POST /api/upload/process
 *
 * Upload audio file and immediately process it
 * This combines upload + processing in one endpoint
 *
 * PROTECTED: Requires valid Auth0 JWT token
 *
 * Form fields:
 * - audio: File (required)
 * - eventName: string (optional)
 * - location: string (optional)
 * - timestamp: string (optional)
 *
 * Headers:
 *   Authorization: Bearer <jwt-token>
 */
router.post('/process', requireAuth, upload.single('audio'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    // Get user ID from authenticated token
    const userId = getUserId(req);

    if (!userId) {
      fs.unlinkSync(req.file.path);
      return res.status(401).json({ error: 'User ID not found in token' });
    }

    const { eventName, location, timestamp } = req.body;

    console.log('[UPLOAD] Processing uploaded audio');
    console.log(`  File: ${req.file.filename}`);
    console.log(`  User: ${userId}`);

    // Use shared processor service with local file path
    const result = await processAndStoreContact({
      audioPath: req.file.path,
      userId: userId,
      eventContext: {
        event_name: eventName,
        location: location,
        timestamp: timestamp
      }
    });

    // Handle result
    if (!result.success) {
      // Clean up file on error
      fs.unlinkSync(req.file.path);

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

    console.log('[UPLOAD] Processing complete');

    // Success response with file info
    res.json({
      success: true,
      contact: result.contact,
      metadata: result.metadata,
      file: {
        filename: req.file.filename,
        size: req.file.size
      }
    });

  } catch (error: any) {
    console.error('[UPLOAD] Unexpected error:', error);

    // Clean up file on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      error: 'Upload and process failed',
      details: error.message
    });
  }
});

/**
 * POST /api/upload/save-from-python
 *
 * Receives contact data from Python service (background callback)
 *
 * Architecture flow:
 * 1. Mobile app uploads directly to Python (fast!)
 * 2. Python returns contact card to mobile (~15s)
 * 3. Python calls THIS endpoint in background
 * 4. We save to database + generate embeddings
 *
 * Form data:
 * - audio: Audio file (optional)
 * - user_id: User ID
 * - contact_data: JSON stringified contact
 * - metadata: JSON stringified metadata
 * - audio_url: Original audio URL (if any)
 */
router.post('/save-from-python', upload.single('audio'), async (req: Request, res: Response) => {
  try {
    console.log('[PYTHON-CALLBACK] Received contact from Python service');

    const { user_id, contact_data, metadata, audio_url } = req.body;
    const audioFile = req.file;

    // Validate required fields
    if (!user_id || !contact_data) {
      return res.status(400).json({
        success: false,
        error: 'user_id and contact_data are required'
      });
    }

    // Parse JSON data
    const contactCard = JSON.parse(contact_data);
    const meta = metadata ? JSON.parse(metadata) : {};

    console.log(`[PYTHON-CALLBACK] Processing contact: ${contactCard.name}`);

    // Determine audio URL (use provided URL or save file locally if needed)
    let finalAudioUrl = audio_url || null;

    if (audioFile && !finalAudioUrl) {
      // Save audio file locally
      const filename = `${Date.now()}-${contactCard.name?.replace(/\s+/g, '-') || 'unknown'}.wav`;
      const savePath = path.join(UPLOAD_DIR, user_id, filename);

      // Ensure user directory exists
      const userDir = path.join(UPLOAD_DIR, user_id);
      if (!fs.existsSync(userDir)) {
        fs.mkdirSync(userDir, { recursive: true });
      }

      fs.writeFileSync(savePath, audioFile.buffer);
      finalAudioUrl = `/uploads/audio/${user_id}/${filename}`;
      console.log(`[PYTHON-CALLBACK] Audio saved locally: ${finalAudioUrl}`);
    }

    // Import necessary modules
    const { db } = await import('../db/index.js');
    const { contacts, conversations } = await import('../db/schema.js');
    const { generateContactEmbedding } = await import('../services/semanticSearch.js');
    const { emitToUser } = await import('../services/socket.js');

    // Save contact to database
    console.log('[DATABASE] Saving contact...');

    const [newContact] = await db.insert(contacts).values({
      userId: user_id,
      name: contactCard.name || null,
      role: contactCard.role || null,
      company: contactCard.company || null,
      location: contactCard.location || null,
      phone: contactCard.phone || null,
      email: contactCard.email || null,
      linkedinUrl: contactCard.linkedin_url || null,
      topicsDiscussed: contactCard.topics_discussed || [],
      followUps: contactCard.follow_ups || [],
      conversationSummary: contactCard.conversation_summary || null,
      confidenceScore: contactCard.confidence_score || null,
      metAt: contactCard.met_at || null,
      metDate: contactCard.met_date ? new Date(contactCard.met_date) : null,
      metLocation: meta.location || null
    }).returning();

    console.log(`[DATABASE] Contact saved with ID: ${newContact.id}`);

    // Save conversation record
    if (meta.raw_data || finalAudioUrl) {
      await db.insert(conversations).values({
        contactId: newContact.id,
        audioUrl: finalAudioUrl || '',
        rawTranscription: meta.raw_data?.full_transcription || null,
        diarizedTranscription: meta.raw_data?.utterances || null,
        conversationSummary: contactCard.conversation_summary,
        durationSeconds: meta.audio_duration || null,
        eventName: meta.event_name || null
      });
      console.log('[DATABASE] Conversation record saved');
    }

    // Generate embedding for semantic search (async, don't wait)
    console.log('[SEMANTIC] Generating embedding in background...');
    generateContactEmbedding(newContact.id).catch(error => {
      console.error('[SEMANTIC] Failed to generate embedding:', error);
    });

    // Notify user via Socket.io (if connected)
    try {
      emitToUser(user_id, 'contact:saved', {
        contact: newContact,
        message: 'Contact saved to your account'
      });
      console.log('[SOCKET] User notified via Socket.io');
    } catch (error) {
      console.log('[SOCKET] User not connected');
    }

    // Return success to Python
    res.json({
      success: true,
      contact_id: newContact.id,
      message: 'Contact saved successfully'
    });

  } catch (error: any) {
    console.error('[PYTHON-CALLBACK] Error saving contact:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Error handler for multer errors
 */
router.use((error: any, req: any, res: any, next: any) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        error: 'File too large',
        details: 'Maximum file size is 100MB'
      });
    }
    return res.status(400).json({
      error: 'File upload error',
      details: error.message
    });
  }

  next(error);
});

export default router;
