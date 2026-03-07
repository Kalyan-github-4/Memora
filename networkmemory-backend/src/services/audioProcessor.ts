/**
 * Audio Processing Service
 *
 * Shared logic for processing audio and storing contacts
 * Used by both /api/audio/process and /api/upload/process
 */

import axios from 'axios';
import { db } from '../db/index.js';
import { contacts, conversations } from '../db/schema.js';
import { callPythonAIService } from './auth0M2M.js';
import { generateContactEmbedding } from './semanticSearch.js';

export interface ProcessAudioOptions {
  audioPath: string;        // Can be URL or local file path
  userId: string;
  eventContext?: {
    event_name?: string;
    location?: string;
    timestamp?: string;
  };
}

export interface ProcessAudioResult {
  success: boolean;
  contact?: any;
  metadata?: {
    processing_time: number;
    audio_duration: number;
    num_speakers: number;
    utterances_count: number;
  };
  error?: string;
  details?: any;
}

/**
 * Process audio file and store contact in database
 *
 * This is the core function used by all audio processing endpoints
 */
export async function processAndStoreContact(
  options: ProcessAudioOptions
): Promise<ProcessAudioResult> {
  const { audioPath, userId, eventContext } = options;

  try {
    console.log('[PROCESSOR] Processing audio');
    console.log(`  Audio: ${audioPath}`);
    console.log(`  User ID: ${userId}`);
    if (eventContext?.event_name) {
      console.log(`  Event: ${eventContext.event_name}`);
    }

    // Call Python AI service with Auth0 M2M authentication
    console.log('[PROCESSOR] Calling Python AI service with Auth0 M2M authentication...');

    const aiResponse = await callPythonAIService('/api/audio/process', {
      audio_url: audioPath,
      event_context: eventContext
    }, 120000);

    // Check for errors from Python service
    if (aiResponse.status === 'error') {
      console.error('[PROCESSOR] Python AI returned error:', aiResponse.error);
      return {
        success: false,
        error: 'AI processing failed',
        details: aiResponse.error
      };
    }

    const contactCard = aiResponse.contact_card;
    const metadata = aiResponse.metadata;

    console.log('[PROCESSOR] AI processing successful');
    console.log(`  Contact: ${contactCard.name || 'Unknown'}`);
    console.log(`  Company: ${contactCard.company || 'Unknown'}`);

    // Store contact in database
    console.log('[PROCESSOR] Storing contact in database...');

    const [newContact] = await db.insert(contacts).values({
      userId: userId,
      name: contactCard.name || null,
      role: contactCard.role || null,
      company: contactCard.company || null,
      location: contactCard.location || null,
      phone: contactCard.phone || null,
      email: contactCard.email || null,
      linkedinUrl: contactCard.linkedin_url || null,
      topicsDiscussed: contactCard.topics_discussed || [],
      followUps: contactCard.follow_ups || [],
      conversationSummary: contactCard.conversation_summary,
      confidenceScore: contactCard.confidence_score,
      metAt: eventContext?.event_name || null,
      metDate: eventContext?.timestamp ? new Date(eventContext.timestamp) : null,
      metLocation: eventContext?.location || null
    }).returning();

    console.log(`[PROCESSOR] Contact stored with ID: ${newContact.id}`);

    // Generate embedding for semantic search (async, don't wait)
    console.log('[PROCESSOR] Generating embedding for semantic search...');
    generateContactEmbedding(newContact.id).catch(error => {
      console.error('[PROCESSOR] Failed to generate embedding:', error);
      // Don't fail the request if embedding generation fails
    });

    // Store conversation record
    console.log('[PROCESSOR] Storing conversation record...');

    await db.insert(conversations).values({
      contactId: newContact.id,
      audioUrl: audioPath,
      rawTranscription: aiResponse.raw_data?.full_transcription || null,
      diarizedTranscription: aiResponse.raw_data?.utterances || null,
      conversationSummary: contactCard.conversation_summary,
      durationSeconds: metadata.audio_duration,
      eventName: eventContext?.event_name || null
    });

    console.log('[PROCESSOR] Processing complete!');

    return {
      success: true,
      contact: newContact,
      metadata: {
        processing_time: metadata.processing_time,
        audio_duration: metadata.audio_duration,
        num_speakers: metadata.num_speakers,
        utterances_count: metadata.utterances_count
      }
    };

  } catch (error: any) {
    console.error('[PROCESSOR] Processing error:', error.message);

    // Check if it's an axios error (Python service call failed)
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        return {
          success: false,
          error: 'Python AI service is not available',
          details: 'Cannot connect to Python service. Is it running?'
        };
      }

      if (error.response) {
        return {
          success: false,
          error: 'Python AI service error',
          details: error.response.data
        };
      }
    }

    // Generic error
    return {
      success: false,
      error: 'Failed to process audio',
      details: error.message
    };
  }
}
