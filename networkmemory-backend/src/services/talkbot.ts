/**
 * TalkBot AI Service
 *
 * THE CORE INNOVATION:
 * User discusses their ideas/problems with AI → AI helps find collaborators from their network
 *
 * Example flow:
 * User: "I want to build a blockchain payment system"
 * AI: "I found Raj from DevFest - he's a blockchain expert at Coinbase!"
 *
 * Uses:
 * - Gemini AI for conversation
 * - Function calling to search contacts semantically
 * - Vector embeddings for similarity matching
 */

import { GoogleGenAI } from '@google/genai';
import { db } from '../db/index.js';
import { chatMessages, contacts } from '../db/schema.js';
import { eq, desc, and } from 'drizzle-orm';
import { searchContactsSemantically } from './semanticSearch.js';

// Initialize Gemini AI with new library
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

// Define the contact search tool for AI using new library format
const searchContactsTool: any = {
  name: 'search_contacts',
  description: `Search the user's network for people with specific skills, expertise, or topics.
  Use this when the user needs help with something or wants to collaborate.
  Returns people from their network who match the query.`,
  parameters: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'What kind of person or expertise to search for (e.g., "blockchain expert", "mobile app developer", "marketing specialist")'
      },
      limit: {
        type: 'number',
        description: 'Maximum number of contacts to return (default: 5)'
      }
    },
    required: ['query']
  }
};

/**
 * Get chat history for user (last N messages)
 */
async function getChatHistory(userId: string, sessionId: string, limit: number = 20) {
  const history = await db
    .select()
    .from(chatMessages)
    .where(and(
      eq(chatMessages.userId, userId),
      eq(chatMessages.sessionId, sessionId)
    ))
    .orderBy(desc(chatMessages.createdAt))
    .limit(limit);

  return history.reverse();  // Oldest first
}

/**
 * Save chat message to database
 */
async function saveChatMessage(
  userId: string,
  sessionId: string,
  role: 'user' | 'assistant',
  content: string,
  toolCalls?: any[],
  toolResults?: any[]
) {
  const [message] = await db.insert(chatMessages).values({
    userId,
    sessionId,
    role,
    content,
    toolCalls: toolCalls || null,
    toolResults: toolResults || null
  }).returning();

  return message;
}

/**
 * Main TalkBot chat function
 *
 * Handles user message, calls AI, executes tool calls, returns response
 */
export async function processChatMessage(
  userId: string,
  sessionId: string,
  userMessage: string
): Promise<{
  success: boolean;
  response?: string;
  toolCalls?: any[];
  error?: string;
}> {
  try {
    console.log(`[TALKBOT] Processing message from user ${userId}`);
    console.log(`  Message: "${userMessage}"`);

    // Save user message
    await saveChatMessage(userId, sessionId, 'user', userMessage);

    // Get conversation history
    const history = await getChatHistory(userId, sessionId);
    console.log(`[TALKBOT] Loaded ${history.length} previous messages`);

    // Prepare system instruction
    const systemInstruction = `You are TalkBot, an AI assistant that helps users brainstorm ideas and find collaborators from their professional network.

**Your Role:**
- Help users discuss and refine their ideas
- When users need help with something specific, search their network for people who can help
- Be conversational and friendly
- Focus on actionable collaboration opportunities

**When to use search_contacts tool:**
- User mentions needing help with something (e.g., "I need a blockchain expert")
- User describes a problem/project that requires specific expertise
- User asks "who can help me with X?"
- User wants to collaborate on something

**Response style:**
- Conversational and helpful
- When you find contacts, explain why they're a good match
- Suggest next steps (e.g., "Want me to draft a message?")`;

    // Build conversation contents
    const contents = [
      { role: 'user', parts: [{ text: systemInstruction }] },
      ...history.map((msg: any) => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }]
      })),
      { role: 'user', parts: [{ text: userMessage }] }
    ];

    // Call Gemini with new API and tools
    const response = await genAI.models.generateContent({
      model: 'gemini-2.5-flash',  // Latest free tier model
      contents: contents,
      config: {
        tools: [
          {
            functionDeclarations: [searchContactsTool]
          }
        ]
      }
    });

    // Check for function calls
    const candidates = response.candidates || [];
    if (candidates.length === 0) {
      throw new Error('No response candidates from AI');
    }

    const candidate = candidates[0];
    const parts = candidate.content?.parts || [];

    // Check if there are function calls
    const functionCalls = parts.filter((part: any) => part.functionCall);

    if (functionCalls.length > 0) {
      console.log(`[TALKBOT] AI requested ${functionCalls.length} tool call(s)`);

      const toolResults: any[] = [];
      const executedCalls: any[] = [];

      for (const part of functionCalls) {
        const call = part.functionCall;
        if (!call) continue;

        console.log(`[TALKBOT] Executing tool: ${call.name}`);
        console.log(`  Args: ${JSON.stringify(call.args)}`);

        if (call.name === 'search_contacts' && call.args) {
          // Execute semantic search
          const query = call.args.query as string;
          const limit = (call.args.limit as number) || 5;

          const searchResults = await searchContactsSemantically(userId, query, limit);

          toolResults.push({
            functionResponse: {
              name: call.name,
              response: {
                contacts: searchResults.map(c => ({
                  name: c.name,
                  role: c.role,
                  company: c.company,
                  topics_discussed: c.topicsDiscussed,
                  met_at: c.metAt,
                  summary: c.conversationSummary
                }))
              }
            }
          });

          executedCalls.push(call);
          console.log(`[TALKBOT] Found ${searchResults.length} matching contacts`);
        }
      }

      // Send tool results back to AI for final response
      const finalResponse = await genAI.models.generateContent({
        model: 'gemini-2.5-flash',  // Same model for consistency
        contents: [
          ...contents,
          {
            role: 'model',
            parts: functionCalls
          },
          {
            role: 'user',
            parts: toolResults
          }
        ]
      });

      const finalText = finalResponse.text || '';
      console.log(`[TALKBOT] Final AI response (with tool results): "${finalText}"`);

      // Save assistant message with tool usage
      await saveChatMessage(userId, sessionId, 'assistant', finalText, executedCalls, toolResults);

      return {
        success: true,
        response: finalText,
        toolCalls: executedCalls
      };

    } else {
      // No tool calls - just return AI response
      const aiResponse = response.text || '';
      console.log(`[TALKBOT] AI response: "${aiResponse}"`);

      // Save assistant message
      await saveChatMessage(userId, sessionId, 'assistant', aiResponse);

      return {
        success: true,
        response: aiResponse
      };
    }

  } catch (error: any) {
    console.error('[TALKBOT] Error processing message:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Create a new chat session
 */
export function generateSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}

/**
 * Get chat session history
 */
export async function getChatSession(userId: string, sessionId: string) {
  const messages = await db
    .select()
    .from(chatMessages)
    .where(and(
      eq(chatMessages.userId, userId),
      eq(chatMessages.sessionId, sessionId)
    ))
    .orderBy(chatMessages.createdAt);

  return messages;
}
