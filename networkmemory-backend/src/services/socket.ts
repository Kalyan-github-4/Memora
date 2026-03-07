/**
 * Socket.io Service
 *
 * Real-time bidirectional communication for:
 * - TalkBot chat (AI conversations)
 * - Audio processing updates (progress notifications)
 * - Contact suggestions (semantic search results)
 * - Live collaboration matching
 */

import { Server as HTTPServer } from 'http';
import { Server as SocketIOServer, Socket } from 'socket.io';
import { auth } from 'express-oauth2-jwt-bearer';

interface AuthenticatedSocket extends Socket {
  userId?: string;
  userType?: string;
  privacyLevel?: string;
}

let io: SocketIOServer | null = null;

/**
 * Initialize Socket.io server
 */
export function initializeSocket(httpServer: HTTPServer): SocketIOServer {
  io = new SocketIOServer(httpServer, {
    cors: {
      origin: process.env.CORS_ORIGIN || '*',
      methods: ['GET', 'POST'],
      credentials: true
    },
    transports: ['websocket', 'polling']
  });

  // Authentication middleware for Socket.io
  io.use(async (socket: AuthenticatedSocket, next) => {
    try {
      const token = socket.handshake.auth.token || socket.handshake.headers.authorization?.replace('Bearer ', '');

      if (!token) {
        // Development mode - allow without token
        if (process.env.NODE_ENV === 'development') {
          console.warn('[SOCKET] Development mode - accepting unauthenticated connection');
          socket.userId = socket.handshake.auth.userId || 'dev-user-123';
          socket.userType = 'individual';
          socket.privacyLevel = 'private';
          return next();
        }
        return next(new Error('Authentication required'));
      }

      // TODO: Validate JWT token with Auth0
      // For now, accept in development mode
      if (process.env.NODE_ENV === 'development') {
        socket.userId = socket.handshake.auth.userId || 'dev-user-123';
        socket.userType = 'individual';
        socket.privacyLevel = 'private';
        return next();
      }

      next();
    } catch (error) {
      console.error('[SOCKET] Authentication error:', error);
      next(new Error('Authentication failed'));
    }
  });

  // Connection handler
  io.on('connection', (socket: AuthenticatedSocket) => {
    const userId = socket.userId;
    console.log(`[SOCKET] User connected: ${userId} (${socket.id})`);

    // Join user-specific room for targeted messages
    if (userId) {
      socket.join(`user:${userId}`);
      console.log(`[SOCKET] User ${userId} joined room: user:${userId}`);
    }

    // TalkBot: User sends message
    socket.on('chat:message', async (data) => {
      console.log(`[SOCKET] Chat message from ${userId}:`, data.message);

      try {
        // Emit acknowledgment
        socket.emit('chat:message:received', { messageId: data.messageId });

        // Process message with AI
        socket.emit('chat:typing', { isTyping: true });

        // Import and call TalkBot service
        const { processChatMessage, generateSessionId } = await import('./talkbot.js');

        const sessionId = data.sessionId || generateSessionId();
        const result = await processChatMessage(userId!, data.message, sessionId);

        socket.emit('chat:typing', { isTyping: false });

        if (result.success) {
          socket.emit('chat:message', {
            role: 'assistant',
            content: result.response,
            toolCalls: result.toolCalls,
            sessionId
          });
        } else {
          socket.emit('chat:error', { error: result.error });
        }

      } catch (error: any) {
        console.error('[SOCKET] Error processing chat message:', error);
        socket.emit('chat:error', { error: 'Failed to process message' });
      }
    });

    // TalkBot: Request contact search
    socket.on('chat:search_contacts', async (data) => {
      console.log(`[SOCKET] Contact search request from ${userId}:`, data.query);

      try {
        // Perform semantic search on contacts
        const { searchContactsSemantically } = await import('./semanticSearch.js');

        const results = await searchContactsSemantically(userId!, data.query, data.limit || 5);

        socket.emit('chat:search_results', {
          query: data.query,
          results: results
        });
      } catch (error: any) {
        console.error('[SOCKET] Error searching contacts:', error);
        socket.emit('chat:error', { error: 'Failed to search contacts' });
      }
    });

    // Audio processing: Subscribe to updates
    socket.on('audio:subscribe', (data) => {
      const { processingId } = data;
      socket.join(`processing:${processingId}`);
      console.log(`[SOCKET] User ${userId} subscribed to processing:${processingId}`);
    });

    // Audio processing: Unsubscribe
    socket.on('audio:unsubscribe', (data) => {
      const { processingId } = data;
      socket.leave(`processing:${processingId}`);
      console.log(`[SOCKET] User ${userId} unsubscribed from processing:${processingId}`);
    });

    // Disconnect handler
    socket.on('disconnect', (reason) => {
      console.log(`[SOCKET] User disconnected: ${userId} (${socket.id}) - ${reason}`);
    });

    // Error handler
    socket.on('error', (error) => {
      console.error(`[SOCKET] Socket error for user ${userId}:`, error);
    });
  });

  console.log('[SOCKET] Socket.io server initialized');
  return io;
}

/**
 * Get Socket.io server instance
 */
export function getIO(): SocketIOServer {
  if (!io) {
    throw new Error('Socket.io not initialized. Call initializeSocket first.');
  }
  return io;
}

/**
 * Emit event to specific user
 */
export function emitToUser(userId: string, event: string, data: any): void {
  if (!io) {
    console.warn('[SOCKET] Socket.io not initialized, cannot emit to user');
    return;
  }

  io.to(`user:${userId}`).emit(event, data);
  console.log(`[SOCKET] Emitted ${event} to user:${userId}`);
}

/**
 * Emit event to processing room (audio processing updates)
 */
export function emitToProcessing(processingId: string, event: string, data: any): void {
  if (!io) {
    console.warn('[SOCKET] Socket.io not initialized, cannot emit to processing room');
    return;
  }

  io.to(`processing:${processingId}`).emit(event, data);
  console.log(`[SOCKET] Emitted ${event} to processing:${processingId}`);
}

/**
 * Broadcast to all connected clients
 */
export function broadcast(event: string, data: any): void {
  if (!io) {
    console.warn('[SOCKET] Socket.io not initialized, cannot broadcast');
    return;
  }

  io.emit(event, data);
  console.log(`[SOCKET] Broadcasted ${event} to all clients`);
}
