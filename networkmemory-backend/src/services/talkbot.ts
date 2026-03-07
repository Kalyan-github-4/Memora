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

import { GoogleGenerativeAI, FunctionDeclaration, Tool, SchemaType } from '@google/generative-ai';
import { db } from '../db/index.js';
import { chatMessages, contacts } from '../db/schema.js';
import { eq, desc, and } from 'drizzle-orm';
import { searchContactsSemantically } from './semanticSearch.js';

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Define the contact search tool for AI
const searchContactsTool: FunctionDeclaration = {
  name: 'search_contacts',
  description: `Search the user's network for people with specific skills, expertise, or topics.
  Use this when the user needs help with something or wants to collaborate.
  Returns people from their network who match the query.`,
  parameters: {
    type: SchemaType.OBJECT,
    properties: {
      query: {
        type: SchemaType.STRING,
        description: 'What kind of person or expertise to search for (e.g., "blockchain expert", "mobile app developer", "marketing specialist")'
      },
      limit: {
        type: SchemaType.NUMBER,
        description: 'Maximum number of contacts to return (default: 5)'
      }
    },
    required: ['query']
  }
};

const tools: Tool[] = [
  {
    functionDeclarations: [searchContactsTool]
  }
];

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

    // Prepare context for AI
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

    // Create Gemini model with tools
    const model = genAI.getGenerativeModel({
      model: 'gemini-1.5-flash',
      tools: tools,
      systemInstruction: systemInstruction
    });

    // Build chat history for Gemini
    const chatHistory = history.map((msg: any) => ({
      role: msg.role === 'user' ? 'user' : 'model',
      parts: [{ text: msg.content }]
    }));

    // Start chat
    const chat = model.startChat({
      history: chatHistory
    });

    // Send user message
    const result = await chat.sendMessage(userMessage);
    const response = result.response;

    // Check if AI wants to call a function
    const functionCalls = response.functionCalls();

    if (functionCalls && functionCalls.length > 0) {
      console.log(`[TALKBOT] AI requested ${functionCalls.length} tool call(s)`);

      const toolResults: any[] = [];

      for (const call of functionCalls) {
        console.log(`[TALKBOT] Executing tool: ${call.name}`);
        console.log(`  Args: ${JSON.stringify(call.args)}`);

        if (call.name === 'search_contacts') {
          // Execute semantic search
          const args = call.args as { query: string; limit?: number };
          const query = args.query;
          const limit = args.limit || 5;

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

          console.log(`[TALKBOT] Found ${searchResults.length} matching contacts`);
        }
      }

      // Send tool results back to AI
      const finalResult = await chat.sendMessage(toolResults);
      const finalResponse = finalResult.response.text();

      console.log(`[TALKBOT] Final AI response (with tool results): "${finalResponse}"`);

      // Save assistant message with tool usage
      await saveChatMessage(userId, sessionId, 'assistant', finalResponse, functionCalls, toolResults);

      return {
        success: true,
        response: finalResponse,
        toolCalls: functionCalls
      };

    } else {
      // No tool calls - just return AI response
      const aiResponse = response.text();
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
